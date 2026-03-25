"""
CREDICOUNCIL — Batch Processing Configuration.

Centralized configuration for batch pipeline execution.
Tune these parameters to optimize throughput vs. rate-limit safety.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: A1 Ingestion (parallel with stagger)
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum concurrent A1 ingestion pipelines.
# Each A1 calls Docling OCR + Gemini extraction (~1-2 LLM calls per customer).
A1_MAX_WORKERS: int = 5

# Stagger delay (seconds) between launching each A1 ingestion.
# Prevents all OCR+extraction requests from hitting the LLM simultaneously.
A1_STAGGER_DELAY: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: A2→A3→A4 Processing (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum concurrent A2→A4 pipelines running simultaneously.
# Each pipeline makes multiple LLM calls (A2 feature synthesis + A4 report).
# Gemini Tier-1 free: 15 RPM, 1500 RPD. Adjust based on tier.
PROCESSING_MAX_WORKERS: int = 5

# Stagger delay (seconds) between launching each A2→A4 pipeline.
# Prevents all pipelines from hitting the LLM simultaneously.
PROCESSING_STAGGER_DELAY: float = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# LLM Call Throttling
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum delay (seconds) between consecutive LLM API calls.
# Applied globally across all threads to respect rate limits.
LLM_INTER_CALL_DELAY: float = 0.5

# Maximum retries for a single LLM call on rate-limit errors (429).
LLM_MAX_RETRIES: int = 3

# Base delay (seconds) for exponential backoff on LLM errors.
# Actual delay: base * 2^attempt (e.g., 1s, 2s, 4s).
LLM_RETRY_BASE_DELAY: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# Batch General
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum number of customers allowed in a single batch.
BATCH_MAX_CUSTOMERS: int = 10
