"""
CREDICOUNCIL — Policy RAG Service (Gemini File Search).

Uses Gemini File Search API to provide grounded policy context
for A4 Report Generator. Replaces Amazon OpenSearch Serverless.

Workflow:
  1. init_policy_store.py creates FileSearchStore + uploads policy docs
  2. PolicyRAGService queries the store during report generation
  3. Returns grounded context + citations for LLM narrative
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized client
_rag_client = None
_FILE_SEARCH_MODEL = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-flash")


def _get_rag_client():
    """Lazy-initialize google.genai Client for RAG."""
    global _rag_client
    if _rag_client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _rag_client = genai.Client(api_key=api_key)
        logger.info("RAG client initialized (google.genai)")
    return _rag_client


class PolicyRAGService:
    """RAG service using Gemini File Search for Vietnamese banking policies.

    Provides grounded policy context for A4 report generation.
    If FileSearchStore is not initialized, returns empty context
    (pipeline still works, just without policy citations).
    """

    def __init__(self):
        self._store_name: str | None = os.getenv("FILE_SEARCH_STORE_NAME")

    # ── Store Management ──────────────────────────────────────────────

    def initialize_store(
        self,
        policy_dir: str,
        store_display_name: str = "credicouncil-policy-store",
    ) -> str:
        """Create FileSearchStore and upload all .md files from policy_dir.

        Args:
            policy_dir: Directory containing policy .md files.
            store_display_name: Display name for the store.

        Returns:
            FileSearchStore resource name (e.g. 'fileSearchStores/xxx').
        """
        client = _get_rag_client()
        policy_path = Path(policy_dir)

        if not policy_path.exists():
            raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

        md_files = sorted(policy_path.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No .md files found in {policy_dir}")

        # 1. Create FileSearchStore
        logger.info(f"Creating FileSearchStore: {store_display_name}")
        store = client.file_search_stores.create(
            config={"display_name": store_display_name}
        )
        store_name = store.name
        logger.info(f"Store created: {store_name}")

        # 2. Upload each policy doc
        operations = []
        for md_file in md_files:
            logger.info(f"Uploading: {md_file.name}")
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(md_file),
                file_search_store_name=store_name,
                config={
                    "display_name": md_file.stem,
                    "mime_type": "text/markdown",
                    "chunking_config": {
                        "white_space_config": {
                            "max_tokens_per_chunk": 512,
                            "max_overlap_tokens": 50,
                        }
                    },
                },
            )
            operations.append((md_file.name, op))

        # 3. Wait for all uploads to complete
        for file_name, op in operations:
            while not op.done:
                time.sleep(3)
                op = client.operations.get(op)
            logger.info(f"  ✅ {file_name} indexed")

        self._store_name = store_name
        logger.info(
            f"FileSearchStore ready: {store_name} "
            f"({len(md_files)} documents indexed)"
        )
        return store_name

    def get_store_name(self) -> str | None:
        """Get current FileSearchStore name."""
        return self._store_name

    def list_stores(self) -> list[dict]:
        """List all available FileSearchStores."""
        client = _get_rag_client()
        stores = []
        for store in client.file_search_stores.list():
            stores.append({
                "name": store.name,
                "display_name": getattr(store, "display_name", ""),
            })
        return stores

    def delete_store(self, store_name: str) -> None:
        """Delete a FileSearchStore by name."""
        client = _get_rag_client()
        client.file_search_stores.delete(name=store_name, config={"force": True})
        logger.info(f"Deleted store: {store_name}")

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """Query policy store for grounded context + citations.

        Args:
            question: Natural language question about policy/regulations.
            store_name: Optional override for store name.

        Returns:
            {
                "context": str,        # Grounded policy text
                "citations": list,     # Source citations
                "has_context": bool,   # Whether context was found
            }
        """
        target_store = store_name or self._store_name

        # Graceful fallback — no store configured
        if not target_store:
            logger.warning(
                "FILE_SEARCH_STORE_NAME not set — "
                "skipping policy RAG (report will work without citations)"
            )
            return {
                "context": "",
                "citations": [],
                "has_context": False,
            }

        try:
            from google.genai import types

            client = _get_rag_client()

            def _rag_call():
                return client.models.generate_content(
                    model=_FILE_SEARCH_MODEL,
                    contents=question,
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[target_store]
                                )
                            )
                        ],
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )

            # Retry with exponential backoff (429, 503, timeout)
            MAX_RETRIES = 3
            last_exc = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = _rag_call()
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_retryable = any(kw in err_str for kw in [
                        "429", "resource_exhausted", "rate",
                        "503", "unavailable", "timeout", "connection",
                    ])
                    if not is_retryable or attempt == MAX_RETRIES - 1:
                        raise
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"RAG API error (attempt {attempt+1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    last_exc = e

            # Extract text
            context_text = response.text or ""

            # Extract citations from grounding metadata
            citations = []
            if (
                response.candidates
                and response.candidates[0].grounding_metadata
            ):
                gm = response.candidates[0].grounding_metadata
                # Extract grounding chunks if available
                if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        citation = {}
                        if hasattr(chunk, "retrieved_context"):
                            ctx = chunk.retrieved_context
                            if hasattr(ctx, "uri"):
                                citation["source"] = ctx.uri
                            if hasattr(ctx, "title"):
                                citation["title"] = ctx.title
                        if hasattr(chunk, "web") and chunk.web:
                            if hasattr(chunk.web, "uri"):
                                citation["source"] = chunk.web.uri
                            if hasattr(chunk.web, "title"):
                                citation["title"] = chunk.web.title
                        if citation:
                            citations.append(citation)

                # Also extract grounding supports for inline citations
                if hasattr(gm, "grounding_supports") and gm.grounding_supports:
                    for support in gm.grounding_supports:
                        if hasattr(support, "segment") and support.segment:
                            seg = support.segment
                            citation_entry = {
                                "text": getattr(seg, "text", ""),
                            }
                            if hasattr(support, "grounding_chunk_indices"):
                                citation_entry["chunk_indices"] = list(
                                    support.grounding_chunk_indices or []
                                )
                            citations.append(citation_entry)

            logger.info(
                f"RAG query returned {len(context_text)} chars, "
                f"{len(citations)} citations"
            )

            return {
                "context": context_text,
                "citations": citations,
                "has_context": bool(context_text.strip()),
            }

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "context": "",
                "citations": [],
                "has_context": False,
            }

    # ── Helper: Build query from customer profile ─────────────────

    @staticmethod
    def build_policy_query(
        credit_score: int,
        risk_band: str,
        pd_pct: float,
        customer_type: str = "INDIVIDUAL",
        thin_file: bool = False,
        dti: float | None = None,
        dscr: float | None = None,
        ltv: float | None = None,
        app_row: dict | None = None,
        llm_feats: dict | None = None,
        shap_values: dict | None = None,
    ) -> str:
        """Build a policy-relevant query from customer risk profile.

        Constructs a targeted Vietnamese query using all available data
        to retrieve the most relevant policy clauses from RAG store.
        """
        app = app_row or {}
        feats = llm_feats or {}
        shap = shap_values or {}

        # ── 1. Customer profile summary ──────────────────────────────
        contract_type = app.get("NAME_CONTRACT_TYPE", "")
        income_type = app.get("NAME_INCOME_TYPE", "")
        loan_purpose = feats.get("loan_purpose_category", "UNCLEAR")

        # Map to Vietnamese for better RAG matching
        contract_vi = {
            "Cash loans": "vay tiền mặt",
            "Revolving loans": "vay tuần hoàn (thẻ tín dụng)",
        }.get(contract_type, contract_type)

        purpose_vi = {
            "PRODUCTION": "sản xuất kinh doanh",
            "CONSUMPTION": "tiêu dùng",
            "INVESTMENT": "đầu tư",
            "REFINANCING": "tái cơ cấu nợ",
            "UNCLEAR": "chưa xác định",
        }.get(str(loan_purpose).upper(), str(loan_purpose))

        income_vi = {
            "Working": "người lao động",
            "Commercial associate": "đối tác thương mại",
            "Pensioner": "người hưu trí",
            "State servant": "công chức nhà nước",
            "Student": "sinh viên",
            "Businessman": "doanh nhân",
        }.get(income_type, income_type)

        parts = [
            f"Hồ sơ vay: Khách hàng {customer_type} — {income_vi}, "
            f"loại vay: {contract_vi}, mục đích: {purpose_vi}.",
        ]

        # Age context (important for term limits)
        days_birth = app.get("DAYS_BIRTH")
        if days_birth and isinstance(days_birth, (int, float)):
            age = abs(int(days_birth)) // 365
            parts.append(f"Tuổi khách hàng: {age}.")

        # Credit score context
        parts.append(
            f"Điểm tín dụng: {credit_score}/850, risk band {risk_band}, "
            f"xác suất vỡ nợ (PD): {pd_pct:.2f}%."
        )

        # ── 2. Financial ratios (trigger specific policy lookups) ─────
        ratio_parts = []
        if dti is not None:
            ratio_parts.append(f"DTI = {dti*100:.1f}%")
            if dti >= 0.50:
                ratio_parts.append("(vượt ngưỡng cảnh báo 50%)")
            elif dti >= 0.40:
                ratio_parts.append("(mức cảnh báo 40-50%)")
        if dscr is not None:
            ratio_parts.append(f"DSCR = {dscr:.2f}")
            if dscr < 1.2:
                ratio_parts.append("(dưới ngưỡng an toàn 1.2)")
        if ltv is not None:
            ratio_parts.append(f"LTV = {ltv*100:.1f}%")
            if ltv > 0.70:
                ratio_parts.append("(vượt ngưỡng 70%)")
        if ratio_parts:
            parts.append("Chỉ số tài chính: " + ", ".join(ratio_parts) + ".")

        # ── 3. Collateral & asset context ────────────────────────────
        collateral_parts = []
        if app.get("FLAG_OWN_REALTY") == "Y":
            collateral_parts.append("sở hữu bất động sản")
        if app.get("FLAG_OWN_CAR") == "Y":
            car_age = app.get("OWN_CAR_AGE")
            collateral_parts.append(
                f"sở hữu xe ô tô ({car_age} năm)" if car_age else "sở hữu xe ô tô"
            )
        if not collateral_parts:
            collateral_parts.append("không có TSBĐ rõ ràng")
        parts.append("Tài sản: " + ", ".join(collateral_parts) + ".")

        # ── 4. Special flags ─────────────────────────────────────────
        if thin_file:
            parts.append(
                "Khách hàng THIN-FILE chưa có lịch sử CIC."
            )

        # Risk flags from LLM semantic extraction
        risk_flags = feats.get("risk_flags") or []
        if risk_flags and isinstance(risk_flags, list):
            # Take top 3 most relevant flags
            flags_text = "; ".join(str(f) for f in risk_flags[:3])
            parts.append(f"Cảnh báo rủi ro: {flags_text}.")

        # ── 5. SHAP-driven policy targeting ──────────────────────────
        # Top SHAP factors tell us WHICH policies are most relevant
        shap_labels = []
        for factor_list in [
            shap.get("top_positive_factors", []),
            shap.get("top_negative_factors", []),
        ]:
            for f in (factor_list or [])[:3]:
                label = f.get("label_vi") or f.get("feature", "")
                if label:
                    shap_labels.append(label)

        if shap_labels:
            parts.append(
                "Các yếu tố SHAP chính ảnh hưởng đến quyết định: "
                + ", ".join(shap_labels[:5]) + "."
            )

        # ── 6. Target request ────────────────────────────────────────
        parts.append(
            "Trích dẫn điều khoản cụ thể từ TT39/2016 về điều kiện cho vay, "
            "TT11/2021 về phân loại nợ và trích lập dự phòng, "
            "quy định CIC về xếp hạng tín dụng, "
            "Basel II/III về hệ số CAR và rủi ro tín dụng, "
            "và khung đánh giá 5C áp dụng cho hồ sơ này."
        )

        return " ".join(parts)

