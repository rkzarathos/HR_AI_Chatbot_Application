import os
import re
import json
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pageindex import PageIndexClient
import pageindex.utils as utils


# ========================
# CONFIG
# ========================

# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except Exception:
#     pass

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

if not PAGEINDEX_API_KEY:
    raise ValueError("PAGEINDEX_API_KEY environment variable is not set.")

DOCUMENTS_DIR = Path(
    os.getenv("DOCUMENTS_PATH", Path.cwd() / "documents")
).resolve()

PAGEINDEX_MANIFEST_DIR = Path(
    os.getenv("PAGEINDEX_MANIFEST_DIR", Path.cwd() / "pageindex-manifest")
).resolve()

PAGEINDEX_MANIFEST_PATH = PAGEINDEX_MANIFEST_DIR / "pageindex_manifest.json"
PAGEINDEX_TREES_DIR = PAGEINDEX_MANIFEST_DIR / "pageindex_trees"

# Page-level metadata workbook.
# Expected sheet columns:
# - Document
# - Page
# - Section Titles / Headings Found
# - Topics Discussed
PAGE_METADATA_XLSX_PATH = Path(
    os.getenv(
        "PAGE_METADATA_XLSX_PATH",
        DOCUMENTS_DIR / "Page Level Document Breakdown.xlsx",
    )
).resolve()

PAGE_METADATA_SHEET_NAME = os.getenv("PAGE_METADATA_SHEET_NAME", "Page Breakdown")

PRINT_TREES_TO_LOG = False

# Hardcoded PageIndex document list.
# This is the source of truth / failsafe for which files are submitted.
# The metadata workbook enriches these documents but does not control the build list.
DOCUMENTS = [
    "2026 Employee Handbook.pdf",
    "Accident Insurance.pdf",
    "BUILDING ACCESS POLICY.pdf",
    "Benefits Packet Overall.pdf",
    "Critical Illness Insurance.pdf",
    "Curative Getting Care for Members.pdf",
    "Curative Lantern Sales and Account Management.pdf",
    "Curative Onboarding Steps.pdf",
    "Curative Pharmacy Need to Know.pdf",
    "Curative Registration.pdf",
    "Dental Insurance High Plan.pdf",
    "Dental Insurance Low Plan.pdf",
    "Dental Insurance Reference Guide.pdf",
    "EAP Services Reference Guide.pdf",
    "ExponentHR 401K Enrollment.pdf",
    "ExponentHR Obtaining Year End Forms - W2 and 1095-C.pdf",
    "ExponentHR Pay Checks and Direct Deposit.pdf",
    "FMLA Claim Submission Checklist.pdf",
    "FMLA Policy.pdf",
    "Fidelity NetBenefits Registration.pdf",
    "Gallagher Team contact information.pdf",
    "HR Frequently Asked Questions.pdf",
    "Hospital Indemnity Insurance.pdf",
    "In-State EPO Plan.pdf",
    "In-State PPO Max Plan.pdf",
    "In-State PPO Plan.pdf",
    "Lively Employee FSA Quickstart Guide.pdf",
    "Long Term Disability Insurance.pdf",
    "OTSL 401K Guidlines.pdf",
    "OTSL Employee Referral Form.pdf",
    "OTSL Performace Management Module.pdf",
    "OTSL Profit Sharing Plan.pdf",
    "Out of State PPO Max Plan.pdf",
    "Out of State PPO Plan.pdf",
    "Out of State PPOx Plan.pdf",
    "Reporting Time in ExponentHR.pdf",
    "Short Term Disability Insurance.pdf",
    "Term Life Insurance.pdf",
    "Vision Insurance Reference Guide.pdf",
    "Voluntary Life EOI Form.pdf",
    "Voluntary Term Life Insurance.pdf",
]


# Throttle every PageIndex API call.
# 150 seconds = 2.5 minutes.
PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS = float(
    os.getenv("PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS", "90")
)

# Readiness polling also goes through the same throttle.
# Keep this at 0 unless you want an additional delay beyond the throttle.
POLL_INTERVAL_SECONDS = float(os.getenv("PAGEINDEX_POLL_INTERVAL_SECONDS", "0"))

# With 150 seconds between readiness checks, 120 polls = up to 5 hours per document.
MAX_POLLS_PER_DOCUMENT = int(os.getenv("PAGEINDEX_MAX_POLLS_PER_DOCUMENT", "120"))


# ========================
# PAGEINDEX API THROTTLE
# ========================

_LAST_PAGEINDEX_API_CALL_AT: Optional[float] = None


def throttle_pageindex_api(next_call_label: str = "PageIndex API call") -> None:
    """
    Enforce at least PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS seconds
    between consecutive PageIndex API calls.

    This intentionally applies to:
    - submit_document
    - is_retrieval_ready
    - get_tree

    That gives a true API-call throttle, not just a delay between documents.
    """
    global _LAST_PAGEINDEX_API_CALL_AT

    now = time.monotonic()

    if _LAST_PAGEINDEX_API_CALL_AT is not None:
        elapsed = now - _LAST_PAGEINDEX_API_CALL_AT
        remaining = PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS - elapsed

        if remaining > 0:
            print(
                f"Throttling before {next_call_label}: "
                f"sleeping {remaining:.1f} seconds "
                f"to keep at least {PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS:.1f}s "
                f"between PageIndex API calls.",
                flush=True,
            )
            time.sleep(remaining)

    _LAST_PAGEINDEX_API_CALL_AT = time.monotonic()


def pageindex_submit_document(client: PageIndexClient, doc_path: Path) -> Any:
    throttle_pageindex_api(f"submit_document({doc_path.name})")
    return client.submit_document(str(doc_path))


def pageindex_is_retrieval_ready(client: PageIndexClient, doc_id: str, doc_name: str) -> bool:
    throttle_pageindex_api(f"is_retrieval_ready({doc_name})")
    return client.is_retrieval_ready(doc_id)


def pageindex_get_tree(client: PageIndexClient, doc_id: str, doc_name: str) -> Any:
    throttle_pageindex_api(f"get_tree({doc_name})")
    return client.get_tree(doc_id, node_summary=True)


# ========================
# METADATA HELPERS
# ========================

def clean_text(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


def normalize_doc_name(value: Any) -> str:
    return os.path.basename(clean_text(value)).strip().lower()


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        return int(float(value))
    except Exception:
        return None


def split_semicolon_values(text: str) -> List[str]:
    values = []
    seen = set()

    for part in clean_text(text).split(";"):
        item = clean_text(part)
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        values.append(item)

    return values


def load_page_metadata(
    metadata_xlsx_path: Path,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load the page-level metadata workbook.

    Returns:
    1. page_metadata:
       Lookup keyed by (normalized_document_name, page_number).

    2. document_metadata:
       Aggregated metadata per document for manifest/routing/debugging.

    The workbook does NOT control which documents are submitted.
    The hardcoded DOCUMENTS list is the build source of truth.
    """
    if not metadata_xlsx_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_xlsx_path}")

    df = pd.read_excel(metadata_xlsx_path, sheet_name=PAGE_METADATA_SHEET_NAME)
    df.columns = [clean_text(c) for c in df.columns]

    required_columns = {
        "Document",
        "Page",
        "Section Titles / Headings Found",
        "Topics Discussed",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata workbook is missing required columns: {sorted(missing_columns)}"
        )

    page_metadata: Dict[Tuple[str, int], Dict[str, Any]] = {}

    doc_section_values: Dict[str, List[str]] = {}
    doc_topic_values: Dict[str, List[str]] = {}
    doc_page_counts: Dict[str, int] = {}
    doc_display_names: Dict[str, str] = {}

    for _, row in df.iterrows():
        document_name = clean_text(row.get("Document"))
        normalized_document_name = normalize_doc_name(document_name)
        page_number = safe_int(row.get("Page"))

        if not document_name or not normalized_document_name:
            continue

        if normalized_document_name not in doc_display_names:
            doc_display_names[normalized_document_name] = document_name

        if page_number is None:
            continue

        section_titles = clean_text(
            row.get("Section Titles / Headings Found"),
            max_chars=1200,
        )
        topics_discussed = clean_text(
            row.get("Topics Discussed"),
            max_chars=1200,
        )

        page_metadata[(normalized_document_name, page_number)] = {
            "document_name": document_name,
            "metadata_page": page_number,
            "section_titles": section_titles,
            "topics_discussed": topics_discussed,
        }

        doc_page_counts[normalized_document_name] = max(
            doc_page_counts.get(normalized_document_name, 0),
            page_number,
        )

        doc_section_values.setdefault(normalized_document_name, [])
        doc_topic_values.setdefault(normalized_document_name, [])

        for section in split_semicolon_values(section_titles):
            if section.lower() not in {s.lower() for s in doc_section_values[normalized_document_name]}:
                doc_section_values[normalized_document_name].append(section)

        for topic in split_semicolon_values(topics_discussed):
            if topic.lower() not in {t.lower() for t in doc_topic_values[normalized_document_name]}:
                doc_topic_values[normalized_document_name].append(topic)

    document_metadata: Dict[str, Dict[str, Any]] = {}

    for normalized_name, display_name in doc_display_names.items():
        sections = doc_section_values.get(normalized_name, [])
        topics = doc_topic_values.get(normalized_name, [])

        document_metadata[normalized_name] = {
            "document_name": display_name,
            "normalized_document_name": normalized_name,
            "metadata_page_count": doc_page_counts.get(normalized_name, 0),
            "metadata_sections": sections[:80],
            "metadata_topics": topics[:120],
            "metadata_sections_text": "; ".join(sections[:80]),
            "metadata_topics_text": "; ".join(topics[:120]),
        }

    print(
        f"Loaded metadata for {len(doc_display_names)} documents and "
        f"{len(page_metadata)} pages from {metadata_xlsx_path}"
    )

    return page_metadata, document_metadata


# ========================
# GENERAL HELPERS
# ========================

def utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_filename(name: str) -> str:
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in name
    )


def extract_doc_id(submit_response: Any) -> str:
    """
    PageIndex cookbook-style response is expected to include:
        {"doc_id": "..."}
    This helper is defensive in case response shape varies.
    """
    if isinstance(submit_response, dict):
        doc_id = (
            submit_response.get("doc_id")
            or submit_response.get("id")
            or submit_response.get("document_id")
        )
    else:
        doc_id = str(submit_response)

    if not doc_id:
        raise RuntimeError(
            f"Could not find doc_id in PageIndex submit response: {submit_response}"
        )

    return doc_id


def wait_until_retrieval_ready(
    client: PageIndexClient,
    doc_id: str,
    doc_name: str,
) -> None:
    """
    Waits until PageIndex says the submitted document is retrieval-ready.

    Readiness checks are also throttled to avoid rapid API calls.
    """
    for attempt in range(1, MAX_POLLS_PER_DOCUMENT + 1):
        try:
            if pageindex_is_retrieval_ready(client, doc_id, doc_name):
                print(f"PageIndex ready: {doc_name}")
                return
        except Exception as e:
            print(f"PageIndex readiness check failed for {doc_name}: {e}")

        print(
            f"Waiting for PageIndex: {doc_name} "
            f"({attempt}/{MAX_POLLS_PER_DOCUMENT})",
            flush=True,
        )

        if POLL_INTERVAL_SECONDS > 0:
            time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out waiting for PageIndex to process {doc_name}. doc_id={doc_id}"
    )


def get_tree_result(
    client: PageIndexClient,
    doc_id: str,
    doc_name: str,
) -> Any:
    """
    Fetches the actual PageIndex tree object.

    Important:
    app.py expects the saved tree file to contain the tree itself,
    not a wrapper like {"status": "...", "result": {...}}.
    """
    tree_response = pageindex_get_tree(client, doc_id, doc_name)

    if isinstance(tree_response, dict) and "result" in tree_response:
        tree = tree_response["result"]
    else:
        tree = tree_response

    if not tree:
        raise RuntimeError(f"Empty PageIndex tree returned for {doc_name}")

    return tree


def save_tree_file(
    doc_name: str,
    doc_id: str,
    tree: Any,
) -> str:
    PAGEINDEX_TREES_DIR.mkdir(parents=True, exist_ok=True)

    tree_file = PAGEINDEX_TREES_DIR / f"{safe_filename(doc_name)}__{doc_id}.json"

    with tree_file.open("w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    return str(tree_file)


def submit_and_save_document(
    client: PageIndexClient,
    doc_path: Path,
    document_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    doc_name = doc_path.name

    print(f"\nSubmitting to PageIndex: {doc_name}")

    submit_response = pageindex_submit_document(client, doc_path)
    doc_id = extract_doc_id(submit_response)

    print(f"Submitted {doc_name}. doc_id={doc_id}")

    wait_until_retrieval_ready(
        client=client,
        doc_id=doc_id,
        doc_name=doc_name,
    )

    tree = get_tree_result(
        client=client,
        doc_id=doc_id,
        doc_name=doc_name,
    )

    if PRINT_TREES_TO_LOG:
        print(f"\nPageIndex tree for {doc_name}:")
        try:
            utils.print_tree(tree)
        except Exception as e:
            print(f"Could not print PageIndex tree for {doc_name}: {e}")

    tree_file = save_tree_file(
        doc_name=doc_name,
        doc_id=doc_id,
        tree=tree,
    )

    print(f"Saved tree file: {tree_file}")

    return {
        "doc_name": doc_name,
        "doc_path": str(doc_path),
        "pageindex_doc_id": doc_id,
        "tree_file": tree_file,
        "submitted_at_utc": utc_now_iso(),
        "api_key_used": "PAGEINDEX_API_KEY",

        # Metadata is stored in the manifest, not injected into the PDF/tree text.
        # Runtime app.py can use this for document routing and source/debug display.
        "metadata": document_metadata or {},
    }


# ========================
# MAIN
# ========================

def main() -> None:
    print("Starting PageIndex build...")
    print(f"DOCUMENTS_DIR: {DOCUMENTS_DIR}")
    print(f"PAGEINDEX_MANIFEST_DIR: {PAGEINDEX_MANIFEST_DIR}")
    print(f"PAGEINDEX_MANIFEST_PATH: {PAGEINDEX_MANIFEST_PATH}")
    print(f"PAGEINDEX_TREES_DIR: {PAGEINDEX_TREES_DIR}")
    print(f"PAGE_METADATA_XLSX_PATH: {PAGE_METADATA_XLSX_PATH}")
    print(f"PAGE_METADATA_SHEET_NAME: {PAGE_METADATA_SHEET_NAME}")
    print(
        "PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS: "
        f"{PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS}"
    )

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"DOCUMENTS_PATH does not exist: {DOCUMENTS_DIR}")

    page_metadata, document_metadata_by_name = load_page_metadata(
        PAGE_METADATA_XLSX_PATH
    )

    PAGEINDEX_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_TREES_DIR.mkdir(parents=True, exist_ok=True)

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    manifest: Dict[str, Any] = {
        "built_at_utc": utc_now_iso(),
        "documents_dir": str(DOCUMENTS_DIR),
        "pageindex_manifest_dir": str(PAGEINDEX_MANIFEST_DIR),
        "pageindex_trees_dir": str(PAGEINDEX_TREES_DIR),
        "page_metadata_xlsx_path": str(PAGE_METADATA_XLSX_PATH),
        "page_metadata_sheet_name": PAGE_METADATA_SHEET_NAME,
        "pageindex_api_min_seconds_between_calls": PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS,
        "documents_requested": len(DOCUMENTS),
        "documents_submitted": 0,
        "documents_missing": [],
        "documents_failed": [],
        "documents_without_metadata": [],
        "documents": [],
        "api_key_used": "PAGEINDEX_API_KEY",
    }

    for doc_index, doc_name in enumerate(DOCUMENTS, start=1):
        doc_path = DOCUMENTS_DIR / doc_name
        normalized_doc_name = normalize_doc_name(doc_name)
        doc_metadata = document_metadata_by_name.get(normalized_doc_name, {})

        if not doc_metadata:
            print(
                f"WARNING: No metadata found in workbook for hardcoded document: {doc_name}",
                flush=True,
            )
            manifest["documents_without_metadata"].append(doc_name)

        print(
            f"\n===== Document {doc_index}/{len(DOCUMENTS)}: {doc_name} =====",
            flush=True,
        )

        if not doc_path.exists():
            print(f"WARNING: Missing document: {doc_path}")
            manifest["documents_missing"].append(
                {
                    "doc_name": doc_name,
                    "doc_path": str(doc_path),
                    "metadata": doc_metadata,
                }
            )
            continue

        if doc_path.suffix.lower() != ".pdf":
            print(f"WARNING: Skipping non-PDF document: {doc_path}")
            manifest["documents_failed"].append(
                {
                    "doc_name": doc_name,
                    "doc_path": str(doc_path),
                    "error": "not_a_pdf",
                    "metadata": doc_metadata,
                }
            )
            continue

        try:
            print(f"Using PAGEINDEX_API_KEY for {doc_name}", flush=True)

            doc_record = submit_and_save_document(
                client=client,
                doc_path=doc_path,
                document_metadata=doc_metadata,
            )

            manifest["documents"].append(doc_record)
            manifest["documents_submitted"] += 1

        except Exception as e:
            print(f"ERROR: Failed PageIndex build for {doc_name}: {e}", flush=True)
            manifest["documents_failed"].append(
                {
                    "doc_name": doc_name,
                    "doc_path": str(doc_path),
                    "error": str(e),
                    "api_key_attempted": "PAGEINDEX_API_KEY",
                    "metadata": doc_metadata,
                }
            )

        # Save progress after every document so a long run has a usable partial manifest.
        with PAGEINDEX_MANIFEST_PATH.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Progress manifest saved to: {PAGEINDEX_MANIFEST_PATH}", flush=True)

    manifest["completed_at_utc"] = utc_now_iso()

    with PAGEINDEX_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nPageIndex build complete.")
    print(f"Documents requested: {manifest['documents_requested']}")
    print(f"Documents submitted: {manifest['documents_submitted']}")
    print(f"Documents missing: {len(manifest['documents_missing'])}")
    print(f"Documents failed: {len(manifest['documents_failed'])}")
    print(f"Manifest saved to: {PAGEINDEX_MANIFEST_PATH}")

    if manifest["documents_failed"]:
        raise RuntimeError(
            "One or more documents failed during PageIndex build. "
            "Check the startup logs and pageindex_manifest.json."
        )


if __name__ == "__main__":
    main()
