"""
CREDICOUNCIL API — WebSocket Routes for Realtime Pipeline.

Provides /ws/process endpoint that runs A2→A3→A4 sequentially and
broadcasts progress events in realtime to the connected frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from credicouncil.api.pipeline import execute_processing_with_events, format_score_response

logger = logging.getLogger(__name__)

router_ws = APIRouter(tags=["WebSocket"])

# Thread pool for running synchronous pipeline agents
_executor = ThreadPoolExecutor(max_workers=2)


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
