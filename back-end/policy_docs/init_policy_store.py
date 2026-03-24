"""
CreditLens — Initialize Policy FileSearchStore.

Run this script once to create and populate the Gemini FileSearchStore
with Vietnamese banking policy documents for RAG.

Usage:
    cd d:\\project\\swinburn_new\\back-end
    conda activate swinburn_hackathon
    python init_policy_store.py

After running, add the store name to .env:
    FILE_SEARCH_STORE_NAME=fileSearchStores/xxxxx
"""

import sys
import os
import io
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from creditlens.services.policy_rag_service import PolicyRAGService

    print("\n" + "=" * 60)
    print("  CREDITLENS — Policy RAG Store Initialization")
    print("=" * 60)

    rag = PolicyRAGService()
    policy_dir = os.path.join(os.path.dirname(__file__), "policy_docs")

    # Check policy docs exist
    if not os.path.exists(policy_dir):
        print(f"  ❌ Policy directory not found: {policy_dir}")
        sys.exit(1)

    md_files = [f for f in os.listdir(policy_dir) if f.endswith(".md")]
    print(f"  Found {len(md_files)} policy documents:")
    for f in sorted(md_files):
        size_kb = os.path.getsize(os.path.join(policy_dir, f)) / 1024
        print(f"    📄 {f} ({size_kb:.1f} KB)")

    # Check for existing stores
    print("\n  Checking existing stores...")
    try:
        stores = rag.list_stores()
        if stores:
            print(f"  Found {len(stores)} existing store(s):")
            for s in stores:
                print(f"    📦 {s['name']} ({s.get('display_name', 'N/A')})")

            # Ask whether to reuse or create new
            existing_policy = [
                s for s in stores
                if "creditlens" in s.get("display_name", "").lower()
                or "policy" in s.get("display_name", "").lower()
            ]
            if existing_policy:
                store = existing_policy[0]
                print(f"\n  ✅ Existing policy store found: {store['name']}")
                print(f"     Display name: {store.get('display_name', 'N/A')}")
                print(f"\n  To use this store, add to .env:")
                print(f"  FILE_SEARCH_STORE_NAME={store['name']}")
                run_test = input("\n  Run test query? (y/n): ").strip().lower()
                if run_test == "y":
                    _run_test_query(rag, store["name"])
                return
    except Exception as e:
        print(f"  ⚠️ Could not list stores: {e}")

    # Create new store
    print("\n  Creating new FileSearchStore...")
    try:
        store_name = rag.initialize_store(
            policy_dir=policy_dir,
            store_display_name="creditlens-policy-store",
        )
    except Exception as e:
        print(f"  ❌ Failed to create store: {e}")
        sys.exit(1)

    print(f"\n  ✅ Store created: {store_name}")
    print(f"  ✅ {len(md_files)} documents indexed")

    # Save to .env instruction
    print("\n" + "=" * 60)
    print("  ADD THIS LINE TO YOUR .env FILE:")
    print(f"  FILE_SEARCH_STORE_NAME={store_name}")
    print("=" * 60)

    # Append to .env if possible
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                env_content = f.read()
            if "FILE_SEARCH_STORE_NAME" not in env_content:
                with open(env_path, "a", encoding="utf-8") as f:
                    f.write(f"\n# Gemini File Search Store (RAG policy docs)\n")
                    f.write(f"FILE_SEARCH_STORE_NAME='{store_name}'\n")
                print(f"  ✅ Automatically added to {env_path}")
            else:
                print(f"  ⚠️ FILE_SEARCH_STORE_NAME already in .env — update manually")
        except Exception as e:
            print(f"  ⚠️ Could not update .env: {e}")

    # Run test query
    _run_test_query(rag, store_name)


def _run_test_query(rag, store_name):
    """Run a test query to verify the store works."""
    print("\n  Running test query...")
    test_query = (
        "Khách hàng cá nhân, điểm tín dụng 672, risk band AA, "
        "DTI 48%, DSCR 1.18. "
        "Quy định về ngưỡng DTI và điều kiện cho vay theo TT39/2016? "
        "Phân loại nhóm nợ theo TT11/2021?"
    )
    print(f"  Query: {test_query[:80]}...")

    result = rag.query(test_query, store_name=store_name)

    if result["has_context"]:
        print(f"  ✅ Response: {len(result['context'])} chars")
        print(f"  ✅ Citations: {len(result['citations'])}")
        # Show first 300 chars of context
        preview = result["context"][:300]
        print(f"\n  Preview:\n  {preview}...")
    else:
        print("  ❌ No context returned — check store and API key")

    print("\n" + "=" * 60)
    print("  INITIALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
