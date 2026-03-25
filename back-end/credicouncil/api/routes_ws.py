"""
CREDICOUNCIL API — WebSocket Routes for Realtime Pipeline.

Provides /ws/process endpoint that runs A2→A3→A4 sequentially and
broadcasts progress events in realtime to the connected frontend.

Provides /ws/batch endpoint for batch processing lifecycle:
  Phase 1: Sequential A1 ingestion with progress
  Phase 2: Confidence gate review (client sends approve/cancel)
  Phase 3: Parallel A2→A4 processing with per-customer progress
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from credicouncil.api.pipeline import (
    execute_processing_with_events,
    execute_batch_ingestion,
    execute_batch_processing_parallel,
    format_score_response,
)

logger = logging.getLogger(__name__)

router_ws = APIRouter(tags=["WebSocket"])

# Thread pool for running synchronous pipeline agents
_executor = ThreadPoolExecutor(max_workers=2)

# Larger pool for batch processing (multiple customers in parallel)
_batch_executor = ThreadPoolExecutor(max_workers=6)


@router_ws.websocket("/ws/process")
async def ws_process_pipeline(websocket: WebSocket):
    """
    WebSocket endpoint for realtime pipeline processing (A2→A3→A4).

    Client sends a JSON message with ProcessRequest fields:
        { customer_id, application_row, raw_texts, thin_file_flag, identity_consistency_flag }

    Server broadcasts events:
        { event: "started",   step: "A2" }
        { event: "completed", step: "A2", data: { features_count: N } }
        { event: "started",   step: "A3" }
        { event: "completed", step: "A3", data: { credit_score, pd_pct, risk_band } }
        { event: "started",   step: "A4" }
        { event: "completed", step: "A4", data: { five_c_total, recommendation } }
        { event: "done",      result: { full ScoreResponse } }
    Or on error:
        { event: "error", message: "..." }
    """
    await websocket.accept()
    logger.info("[WS] Client connected to /ws/process")

    try:
        # Wait for client to send ProcessRequest data
        raw = await websocket.receive_text()
        params = json.loads(raw)
        logger.info(f"[WS] Received process request for customer: {params.get('customer_id')}")

        # Create an async queue to bridge sync pipeline → async WebSocket
        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def emit_event(event: dict):
            """Called from sync thread to push event to async queue."""
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        def run_pipeline_sync():
            """Run the synchronous pipeline in a thread."""
            try:
                result = execute_processing_with_events(
                    customer_id=params["customer_id"],
                    application_row=params["application_row"],
                    raw_texts=params.get("raw_texts", {}),
                    thin_file_flag=params.get("thin_file_flag", False),
                    identity_consistency_flag=params.get("identity_consistency_flag", "OK"),
                    event_callback=emit_event,
                )
                formatted = format_score_response(result, params["customer_id"])
                metrics = result.pop("__metrics", {})
                emit_event({"event": "done", "result": formatted, "metrics": metrics})
            except Exception as e:
                logger.error(f"[WS] Pipeline error: {e}")
                traceback.print_exc()
                emit_event({"event": "error", "message": str(e)})

        # Launch pipeline in background thread
        future = loop.run_in_executor(_executor, run_pipeline_sync)

        # Forward events from queue to WebSocket until "done" or "error"
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)
            logger.info(f"[WS] Sent event: {event.get('event')} / {event.get('step', '')}")

            if event["event"] in ("done", "error"):
                break

        # Ensure thread finishes
        await future

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Unexpected error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# /ws/batch — Batch processing lifecycle (A1 → CG → A2→A4)
# ═══════════════════════════════════════════════════════════════════════════════


@router_ws.websocket("/ws/batch")
async def ws_batch_pipeline(websocket: WebSocket):
    """
    WebSocket endpoint for full batch pipeline lifecycle.

    ── Phase 1: Client sends initial request ──
    Client → { "action": "start", "customer_ids": ["001", "002", ...] }
    Server → batch_started, a1_started, a1_completed (per customer), phase_ingestion_done

    ── Phase 2: Confidence gate review ──
    Server waits. Client sends one of:
      { "action": "approve", "customer_id": "001", "application_row": {...}, "metadata": {...} }
      { "action": "cancel", "customer_id": "001" }
      { "action": "start_processing" }  ← after all gates handled

    ── Phase 3: Processing ──
    Server → processing_started, customer_step, customer_done, batch_progress, batch_done
    """
    await websocket.accept()
    logger.info("[WS-Batch] Client connected to /ws/batch")

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit_event(event: dict):
        """Thread-safe: push event to async queue."""
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    async def send_events_until(terminal_events: set[str]):
        """Forward events from queue to WebSocket until a terminal event."""
        while True:
            event = await event_queue.get()
            try:
                await websocket.send_json(event)
            except Exception:
                break
            evt_type = event.get("event", "")
            logger.info(f"[WS-Batch] Sent: {evt_type} {event.get('customer_id', '')}")
            if evt_type in terminal_events:
                return event
        return None

    try:
        # ── Wait for initial request ──
        raw = await websocket.receive_text()
        msg = json.loads(raw)

        if msg.get("action") != "start":
            await websocket.send_json({"event": "error", "message": "Expected action=start"})
            return

        customer_ids = msg.get("customer_ids", [])
        if not customer_ids:
            await websocket.send_json({"event": "error", "message": "No customer_ids provided"})
            return

        logger.info(f"[WS-Batch] Starting batch for {len(customer_ids)} customers: {customer_ids}")

        # Track processing time excluding review (Phase 1 + Phase 3 only)
        phase1_start = time.time()

        # ══════════════════════════════════════════════════════════════════
        # Phase 1: Sequential A1 ingestion (in background thread)
        # ══════════════════════════════════════════════════════════════════

        def run_ingestion_phase():
            return execute_batch_ingestion(customer_ids, event_callback=emit_event)

        ingestion_future = loop.run_in_executor(_batch_executor, run_ingestion_phase)

        # Forward all events until phase_ingestion_done
        await send_events_until({"phase_ingestion_done", "error"})

        # Get ingestion results
        ingestion_results = await ingestion_future
        phase1_elapsed = time.time() - phase1_start
        logger.info(f"[WS-Batch] Phase 1 complete: {len(ingestion_results)} results in {phase1_elapsed:.1f}s")

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: Wait for confidence gate decisions from client
        # (review time is NOT counted in total_time)
        # ══════════════════════════════════════════════════════════════════

        approved_customers = []
        pending_reviews = {
            r["customer_id"]
            for r in ingestion_results
            if r["status"] == "OK"
        }

        await websocket.send_json({
            "event": "phase_review_start",
            "pending_count": len(pending_reviews),
            "customer_ids": list(pending_reviews),
        })

        # Listen for approve/cancel/start_processing messages
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "")

            if action == "approve":
                cid = msg["customer_id"]
                pending_reviews.discard(cid)
                approved_customers.append({
                    "customer_id": cid,
                    "application_row": msg["application_row"],
                    "raw_texts": msg.get("metadata", {}).get("raw_texts", {}),
                    "thin_file_flag": msg.get("metadata", {}).get("thin_file_flag", False),
                    "identity_consistency_flag": msg.get("metadata", {}).get(
                        "identity_consistency_flag", "OK"
                    ),
                })
                await websocket.send_json({
                    "event": "review_ack",
                    "customer_id": cid,
                    "action": "approved",
                    "remaining": len(pending_reviews),
                })
                logger.info(f"[WS-Batch] Approved: {cid}, remaining: {len(pending_reviews)}")

            elif action == "cancel":
                cid = msg["customer_id"]
                pending_reviews.discard(cid)
                await websocket.send_json({
                    "event": "review_ack",
                    "customer_id": cid,
                    "action": "cancelled",
                    "remaining": len(pending_reviews),
                })
                logger.info(f"[WS-Batch] Cancelled: {cid}, remaining: {len(pending_reviews)}")

            elif action == "start_processing":
                logger.info(f"[WS-Batch] Start processing: {len(approved_customers)} approved")
                break

            else:
                await websocket.send_json({
                    "event": "error",
                    "message": f"Unknown action: {action}",
                })

        # ══════════════════════════════════════════════════════════════════
        # Phase 3: Parallel A2→A4 processing
        # ══════════════════════════════════════════════════════════════════

        if not approved_customers:
            # No approved customers → batch done immediately
            await websocket.send_json({
                "event": "batch_done",
                "summary": {
                    "total_time_seconds": round(phase1_elapsed, 2),
                    "total_tokens": 0,
                    "total_customers": 0,
                    "success_count": 0,
                    "failed_count": 0,
                    "customers": [],
                },
            })
        else:
            _phase1_elapsed = phase1_elapsed  # capture for closure

            def run_processing_phase():
                result = execute_batch_processing_parallel(
                    approved_customers, event_callback=emit_event
                )
                # total_time = Phase 1 + Phase 3 (excludes Phase 2 review time)
                phase3_elapsed = result.get("summary", {}).get("total_time_seconds", 0)
                total_processing_time = round(_phase1_elapsed + phase3_elapsed, 2)
                if "summary" in result:
                    result["summary"]["total_time_seconds"] = total_processing_time
                return result

            processing_future = loop.run_in_executor(
                _batch_executor, run_processing_phase
            )

            # Forward all events until batch_done
            await send_events_until({"batch_done", "error"})

            # Ensure thread completes
            await processing_future

        logger.info("[WS-Batch] Batch pipeline complete")

    except WebSocketDisconnect:
        logger.info("[WS-Batch] Client disconnected")
    except Exception as e:
        logger.error(f"[WS-Batch] Unexpected error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
