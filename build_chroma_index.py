import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Azure Document Intelligence imports ---
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ========================
# ENV / CONFIG
# ========================

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_PATH", os.path.join(os.getcwd(), "documents"))
CHROMA_DB_PATH = os.getenv("CHROMADB_PATH", "/chromadb")

# Page-level metadata workbook.
# Expected sheet columns:
# - Document
# - Page
# - Section Titles / Headings Found
# - Topics Discussed
PAGE_METADATA_XLSX_PATH = os.getenv(
    "PAGE_METADATA_XLSX_PATH",
    os.path.join(DOCUMENTS_DIR, "Page Level Document Breakdown.xlsx"),
)
PAGE_METADATA_SHEET_NAME = os.getenv("PAGE_METADATA_SHEET_NAME", "Page Breakdown")

# OCR-only build settings.
# This script intentionally skips PDFMiner/PyMuPDF and sends every PDF to Azure OCR.
AZURE_OCR_MODEL_ID = os.getenv("AZURE_OCR_MODEL_ID", "prebuilt-read")

# Keep low enough for F0/free tier safety. 4 seconds = max 15 calls/min.
AZURE_OCR_MIN_SECONDS_BETWEEN_CALLS = float(
    os.getenv("AZURE_OCR_MIN_SECONDS_BETWEEN_CALLS", "4")
)

# Azure Document Intelligence env vars
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

if not AZURE_DOC_INTELLIGENCE_ENDPOINT or not AZURE_DOC_INTELLIGENCE_KEY:
    raise ValueError(
        "Azure Document Intelligence endpoint/key env vars are not set "
        "(AZURE_DOC_INTELLIGENCE_ENDPOINT, AZURE_DOC_INTELLIGENCE_KEY)."
    )

doc_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY),
)

print("Chroma build mode: Azure OCR only")
print(f"Documents path: {DOCUMENTS_DIR}")
print(f"Chroma DB path: {CHROMA_DB_PATH}")
print(f"Metadata workbook: {PAGE_METADATA_XLSX_PATH}")
print(f"Azure OCR model: {AZURE_OCR_MODEL_ID}")
print(f"Azure OCR throttle: {AZURE_OCR_MIN_SECONDS_BETWEEN_CALLS}s between calls")
print(f"Reset Chroma DB: {RESET_CHROMA_DB}")

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

# ========================
# METADATA HELPERS
# ========================

def clean_text(value: Any, max_chars: Optional[int] = None) -> str:
    """
    Normalize metadata cell values into Chroma-safe strings.
    """
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
    """
    Normalize document names so Excel rows match PDF filenames consistently.
    """
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


def load_page_metadata(metadata_xlsx_path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Load only the metadata fields we want to keep on each LangChain/Chroma chunk.

    Used columns:
    - Document
    - Page
    - Section Titles / Headings Found
    - Topics Discussed

    Ignored columns:
    - Total Pages
    - Extraction Notes
    - Text Preview
    """
    if not os.path.exists(metadata_xlsx_path):
        print(f"WARNING: Metadata workbook not found: {metadata_xlsx_path}")
        return {}

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

    metadata_by_page: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for _, row in df.iterrows():
        document_name = clean_text(row.get("Document"))
        normalized_document_name = normalize_doc_name(document_name)
        page_number = safe_int(row.get("Page"))

        if not normalized_document_name or page_number is None:
            continue

        metadata_by_page[(normalized_document_name, page_number)] = {
            "document_name": document_name,
            "metadata_page": page_number,
            "section_titles": clean_text(
                row.get("Section Titles / Headings Found"),
                max_chars=1200,
            ),
            "topics_discussed": clean_text(
                row.get("Topics Discussed"),
                max_chars=1200,
            ),
        }

    print(
        f"Loaded page metadata for {len(metadata_by_page)} pages "
        f"from {metadata_xlsx_path}"
    )

    return metadata_by_page


def get_loader_page_number(doc: Document) -> Optional[int]:
    """
    Different loaders can report page numbers differently.

    PDFMiner/PyMuPDF often use zero-based page indexes.
    Azure Document Intelligence uses one-based page numbers.
    """
    raw_page = doc.metadata.get("page")

    if raw_page is None:
        raw_page = doc.metadata.get("page_number")

    return safe_int(raw_page)


def find_page_metadata(
    doc: Document,
    doc_name: str,
    page_metadata: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Match a loaded LangChain page document to its page-level Excel metadata.

    Tries raw page, raw+1, and raw-1 because loaders differ in page numbering.
    """
    normalized_document_name = normalize_doc_name(doc_name)
    loader_page = get_loader_page_number(doc)

    if loader_page is None:
        return {}

    candidate_pages = [loader_page, loader_page + 1, loader_page - 1]

    seen = set()
    for candidate_page in candidate_pages:
        if candidate_page is None or candidate_page < 1 or candidate_page in seen:
            continue

        seen.add(candidate_page)

        match = page_metadata.get((normalized_document_name, candidate_page))
        if match:
            return match

    return {}


def attach_page_metadata(
    docs_for_file: List[Document],
    doc_name: str,
    page_metadata: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Document]:
    """
    Attach page-level metadata to LangChain Documents.

    IMPORTANT:
    This function does NOT inject metadata into page_content.
    Metadata stays structured only, so it does not increase embedding tokens,
    chunk size, or chunk count.
    """
    enriched_docs: List[Document] = []

    for doc in docs_for_file:
        matched_metadata = find_page_metadata(
            doc=doc,
            doc_name=doc_name,
            page_metadata=page_metadata,
        )

        enriched_metadata = dict(doc.metadata or {})

        # Always retain document identity, even if the Excel metadata row is missing.
        enriched_metadata["document_name"] = doc_name
        enriched_metadata["normalized_document_name"] = normalize_doc_name(doc_name)

        if matched_metadata:
            enriched_metadata.update(matched_metadata)
            enriched_metadata["has_page_metadata"] = True
        else:
            enriched_metadata["metadata_page"] = get_loader_page_number(doc) or ""
            enriched_metadata["section_titles"] = ""
            enriched_metadata["topics_discussed"] = ""
            enriched_metadata["has_page_metadata"] = False

        enriched_docs.append(
            Document(
                page_content=doc.page_content or "",
                metadata=enriched_metadata,
            )
        )

    return enriched_docs


def clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata should stay primitive: str/int/float/bool/None.
    """
    cleaned = {}

    for key, value in (metadata or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = clean_text(value)

    return cleaned


# ========================
# AZURE OCR HELPERS
# ========================

_last_azure_ocr_call_ts = 0.0


def wait_for_azure_ocr_rate_limit() -> None:
    """
    Simple client-side throttle to avoid hitting Azure Document Intelligence
    calls-per-minute limits during index builds.
    """
    global _last_azure_ocr_call_ts

    elapsed = time.time() - _last_azure_ocr_call_ts
    wait_seconds = AZURE_OCR_MIN_SECONDS_BETWEEN_CALLS - elapsed

    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s before next Azure OCR call...")
        time.sleep(wait_seconds)


def azure_ocr_to_documents(file_path: str) -> List[Document]:
    """
    OCR the entire PDF using Azure AI Document Intelligence and return one
    LangChain Document per visual page.

    Important:
    - Azure Document Intelligence page.page_number is 1-based.
    - This preserves the same page model as the page-level metadata workbook.
    """
    global _last_azure_ocr_call_ts

    wait_for_azure_ocr_rate_limit()

    with open(file_path, "rb") as f:
        poller = doc_client.begin_analyze_document(
            model_id=AZURE_OCR_MODEL_ID,
            body=f,
        )

    _last_azure_ocr_call_ts = time.time()

    result = poller.result()

    docs: List[Document] = []

    for page in result.pages:
        lines = [line.content for line in (page.lines or [])]
        page_text = "\n".join(lines).strip()

        docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "source": file_path,
                    "page": page.page_number,
                    "ocr_provider": "azure_document_intelligence",
                    "ocr_model_id": AZURE_OCR_MODEL_ID,
                },
            )
        )

    return docs


def log_metadata_misses_for_file(
    doc_name: str,
    docs_for_file: List[Document],
    page_metadata: Dict[Tuple[str, int], Dict[str, Any]],
    max_examples: int = 5,
) -> None:
    """
    Print a few useful diagnostics when page metadata does not match.
    """
    misses = [
        d for d in docs_for_file
        if not (d.metadata or {}).get("has_page_metadata")
    ]

    if not misses:
        return

    normalized_doc_name = normalize_doc_name(doc_name)
    available_pages = sorted(
        page
        for (doc_key, page) in page_metadata.keys()
        if doc_key == normalized_doc_name
    )

    print(
        f"  Metadata miss diagnostic for {doc_name}: "
        f"Excel pages for this doc={available_pages[:20]}"
        f"{'...' if len(available_pages) > 20 else ''}"
    )

    for d in misses[:max_examples]:
        print(
            f"  Miss example: doc={doc_name}, "
            f"ocr_page={d.metadata.get('page')}, "
            f"text_preview={(d.page_content or '')[:80]!r}"
        )



# ========================
# LOAD & BUILD DATASOURCE
# ========================

page_metadata = load_page_metadata(PAGE_METADATA_XLSX_PATH)

datasource: List[Document] = []
metadata_matches = 0
metadata_misses = 0

for doc_name in DOCUMENTS:
    doc_path = os.path.join(DOCUMENTS_DIR, doc_name)

    if not os.path.exists(doc_path):
        print(f"WARNING: {doc_path} not found")
        continue

    try:
        print(f"OCR processing: {doc_name}")
        docs_for_file = azure_ocr_to_documents(doc_path)
    except Exception as e:
        print(f"Azure Document Intelligence failed for {doc_path}: {e}")
        docs_for_file = []

    if not docs_for_file:
        print(f"Skipping {doc_path}: Azure OCR returned no pages/text.")
        continue

    docs_for_file = attach_page_metadata(
        docs_for_file=docs_for_file,
        doc_name=doc_name,
        page_metadata=page_metadata,
    )

    file_matches = sum(1 for d in docs_for_file if d.metadata.get("has_page_metadata"))
    file_misses = len(docs_for_file) - file_matches

    metadata_matches += file_matches
    metadata_misses += file_misses

    print(
        f"{doc_name}: OCR loaded {len(docs_for_file)} pages "
        f"({file_matches} metadata matches, {file_misses} metadata misses)"
    )

    if file_misses:
        log_metadata_misses_for_file(
            doc_name=doc_name,
            docs_for_file=docs_for_file,
            page_metadata=page_metadata,
        )

    datasource.extend(docs_for_file)

print(f"Collected {len(datasource)} OCR documents/pages")
print(f"Metadata match summary: {metadata_matches} matched, {metadata_misses} missed")

if not datasource:
    raise RuntimeError("No OCR text was extracted. Chroma index was not built.")


# ========================
# SPLIT, EMBED, INDEX
# ========================

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1200,
    chunk_overlap=400,
)

docs = text_splitter.split_documents(datasource)
print(f"Split into {len(docs)} chunks")

# LangChain's splitter copies parent metadata onto child chunks.
# We clean it after splitting so every persisted Chroma chunk has simple metadata.
for d in docs:
    d.metadata = clean_metadata_for_chroma(d.metadata)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH,
)

vectorstore.persist()
print(f"Index built and persisted to {CHROMA_DB_PATH}")
