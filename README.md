# OTSL HR AI Chatbot Application

An HR-focused retrieval augmented generation application for On-Target Supplies & Logistics (OTSL). The app answers employee HR questions from approved HR documents, returns supporting source excerpts, generates answer audio, and records chat, feedback, and survey data for review.

## What This App Does

- Serves a custom web chatbot UI through FastAPI.
- Answers HR questions using only retrieved HR document evidence.
- Runs two retrieval strategies:
  - Chroma vector search with MMR retrieval and CrossEncoder reranking.
  - PageIndex tree search with LLM-based section and node selection.
- Chooses PageIndex when its confidence is high enough, otherwise falls back to Chroma.
- Classifies each question into HR topic categories such as benefits, PTO, leave, payroll, taxes, HR systems, conduct, performance, and retirement.
- Generates a follow-up question for the user.
- Shows source excerpts in the UI.
- Generates MP3 answer audio with Google Text-to-Speech.
- Stores chat logs, classifications, source metadata, feedback, and survey responses in Azure Table Storage.
- Includes offline index builders for Chroma and PageIndex.

## Repository Layout

```text
.
|-- app.py
|-- build_chroma_index.py
|-- build_pageindex.py
|-- Dockerfile
|-- index.html
|-- requirements.txt
|-- startup.sh
`-- data/
    |-- Logo.png
    |-- doc1.pdf
    |-- doc2.pdf
    `-- ...
```

### Main Files

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI backend, retrieval orchestration, LLM prompting, audio generation, Azure Table logging, and API routes. |
| `index.html` | Full frontend UI, including chat input, answer display, source panel, voice input, feedback sidebar, terms modal, and survey modal. |
| `build_chroma_index.py` | Builds the Chroma vector index from HR PDFs using Azure Document Intelligence OCR and OpenAI embeddings. |
| `build_pageindex.py` | Builds PageIndex document trees and a runtime manifest for tree-based retrieval. |
| `Dockerfile` | Container build for running the FastAPI app with Uvicorn. |
| `startup.sh` | Convenience script that runs both index builders. It is not currently used by the Dockerfile `CMD`. |
| `requirements.txt` | Python package dependencies. |

## Runtime Architecture

The application has three main layers:

1. **Frontend**
   - Served from `index.html`.
   - Calls backend endpoints with `fetch`.
   - Stores a browser session ID in `localStorage`.
   - Tracks per-session question count and survey completion in `sessionStorage`.

2. **Backend API**
   - FastAPI application in `app.py`.
   - Serves the frontend, media assets, generated audio, and JSON API responses.
   - Uses LangChain chains and prompts for retrieval routing and answer generation.

3. **Retrieval and Storage**
   - Chroma stores embedded document chunks.
   - PageIndex stores document tree manifests.
   - Azure Table Storage stores chat logs, feedback, and survey responses.
   - Azure Document Intelligence is used by the Chroma build script for OCR.

## Main API Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Serves the chatbot frontend. |
| `/ask` | `POST` | Accepts a user question, retrieves evidence, generates an answer, logs the result, and returns answer/audio/source data. |
| `/feedback` | `POST` | Saves user feedback for a previously logged chat answer. |
| `/survey` | `POST` | Saves the required five-question employee experience survey. |
| `/audio/{filename}` | `GET` | Serves generated MP3 answer audio. |
| `/logo` | `GET` | Serves the company logo from `DOCUMENTS_PATH`. |
| `/dash-logo` | `GET` | Serves the DASH logo from `DOCUMENTS_PATH`. |
| `/thinking-gif` | `GET` | Serves the thinking animation from `DOCUMENTS_PATH`. |

## Retrieval Flow

When a user submits a question to `/ask`:

1. The backend starts PageIndex retrieval and Chroma retrieval at the same time.
2. Chroma retrieval:
   - Uses the persisted Chroma index.
   - Uses MMR retrieval with `k=16`, `fetch_k=60`, and `lambda_mult=0.75`.
   - Reranks candidate chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. PageIndex retrieval:
   - Loads local PageIndex trees from `pageindex_manifest.json`.
   - Uses an LLM to select likely top-level sections.
   - Uses another LLM pass to select exact answer-bearing nodes.
   - Builds evidence context from selected nodes.
4. If PageIndex finishes with confidence above `0.90`, it can be selected.
5. Otherwise the app uses Chroma retrieval as the fallback.
6. The selected context is sent to the answer prompt.
7. The answer prompt returns strict JSON containing the answer, follow-up question, topic classification, confidence, and clarification flag.
8. The backend logs the interaction and returns the answer payload to the frontend.

## Frontend Features

The frontend is a single static HTML file with embedded CSS and JavaScript.

Features include:

- Terms and Conditions modal before chatbot use.
- Dark/light mode toggle.
- Collapsible and pinnable top menu.
- Text chat input.
- Browser speech recognition for voice input.
- Thinking animation while waiting for an answer.
- Markdown-rendered answer display.
- Generated follow-up question button.
- Audio playback for text-to-speech answer output.
- Document source excerpt panel.
- Local chat history sidebar.
- Feedback sidebar.
- Required 1-5 rating survey after two questions.

## Environment Variables

The app requires external services. Configure these variables before building indexes or running the server.

### Required For The App

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Used for embeddings and chat model calls. |
| `AZURE_STORAGE_CONNECTION_STRING` | Used for Azure Blob/Table clients and chat/survey logging. |

### Required For Chroma Index Builds

| Variable | Purpose |
| --- | --- |
| `AZURE_DOC_INTELLIGENCE_ENDPOINT` | Azure Document Intelligence endpoint for OCR. |
| `AZURE_DOC_INTELLIGENCE_KEY` | Azure Document Intelligence API key. |
| `OPENAI_API_KEY` | Used to create OpenAI embeddings for Chroma. |

### Required For PageIndex Builds

| Variable | Purpose |
| --- | --- |
| `PAGEINDEX_API_KEY` | Used by `build_pageindex.py` to submit PDFs and fetch tree data. |

### Optional Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Uvicorn port when running `app.py` directly. |
| `DOCUMENTS_PATH` | `./documents` | Directory containing source documents and frontend image/video assets. |
| `AUDIO_PATH` | `./audio` | Directory where generated MP3 files are written. |
| `CHROMADB_PATH` | `/chromadb` | Chroma persistence directory. |
| `PAGEINDEX_MANIFEST_DIR` | `/pageindex-manifest` in `app.py`; `./pageindex-manifest` in the builder | Directory containing `pageindex_manifest.json` and saved trees. |
| `PAGEINDEX_RUNTIME_MANIFEST_DIR` | `/app/pageindex-manifest` | Runtime path saved into the PageIndex manifest. |
| `CHAT_TABLE_NAME` | `chathistory` | Azure Table name for chat logs. |
| `SURVEY_TABLE_NAME` | `hrchatbotsurvey` | Azure Table name for survey responses. |
| `PAGE_METADATA_XLSX_PATH` | `DOCUMENTS_PATH/Page Level Document Breakdown.xlsx` | Page-level metadata workbook used by indexing scripts. |
| `PAGE_METADATA_SHEET_NAME` | `Page Breakdown` | Metadata worksheet name. |
| `AZURE_OCR_MODEL_ID` | `prebuilt-read` | Azure Document Intelligence OCR model. |
| `AZURE_OCR_MIN_SECONDS_BETWEEN_CALLS` | `4` | Chroma build OCR throttle. |
| `PAGEINDEX_API_MIN_SECONDS_BETWEEN_CALLS` | `20` | PageIndex API throttle. |
| `PAGEINDEX_POLL_INTERVAL_SECONDS` | `0` | Delay between PageIndex readiness polls. |
| `PAGEINDEX_MAX_POLLS_PER_DOCUMENT` | `20` | Max readiness polls per PageIndex document. |
| `BENEFITS_SUMMARY_URL` | BriteHR benefits URL in `app.py` | Link appended to benefits-related answers. |

## Local Setup

Use Python 3.9 or newer. The Dockerfile currently uses Python 3.9.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Source Document Setup

The code expects a document directory controlled by `DOCUMENTS_PATH`. By default, that path is:

```text
./documents
```

The indexing scripts currently use a hardcoded list of HR PDF filenames such as:

- `2026 Employee Handbook.pdf`
- `Benefits Packet Overall.pdf`
- `FMLA Policy.pdf`
- `HR Frequently Asked Questions.pdf`
- `ExponentHR 401K Enrollment.pdf`

The same directory is also expected to contain:

- `Page Level Document Breakdown.xlsx`
- `Logo.png`
- `dash image without bg.png`
- `thinking-gif.mp4`

The repository currently includes a `data/` folder with generic `doc1.pdf` through `doc8.pdf` files and `Logo.png`. To run the app without code changes, provide the expected production files under `DOCUMENTS_PATH`, or update the document path and filename assumptions in the code.

## Building The Indexes

Build Chroma first:

```bash
python build_chroma_index.py
```

Build PageIndex:

```bash
python build_pageindex.py
```

Or run both through:

```bash
bash startup.sh
```

The builders are designed for long-running external API work. They include throttling for Azure Document Intelligence and PageIndex calls.

## Running Locally

After environment variables and indexes are ready:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python app.py
```

Then open:

```text
http://localhost:8000
```

## Docker

Build the image:

```bash
docker build -t otsl-hr-chatbot .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=... \
  -e AZURE_STORAGE_CONNECTION_STRING=... \
  -e DOCUMENTS_PATH=/app/documents \
  -e CHROMADB_PATH=/chromadb \
  -e PAGEINDEX_MANIFEST_DIR=/app/pageindex-manifest \
  otsl-hr-chatbot
```

For production, mount or bake in the prepared document directory, Chroma database, and PageIndex manifest directory.

## Azure Storage Logging

The backend writes to Azure Table Storage:

- Chat history table: `CHAT_TABLE_NAME`, default `chathistory`.
- Survey table: `SURVEY_TABLE_NAME`, default `hrchatbotsurvey`.

Chat records include:

- Session ID
- Row key
- Question
- Answer
- Feedback
- Selected retrieval index
- Topic classification
- Source metadata JSON
- Timestamps

Survey records include:

- Session ID
- Row key
- Five 1-5 question ratings
- Timestamp

## Answering And Safety Rules

The prompt in `app.py` enforces several HR-specific constraints:

- Answer only from retrieved HR materials.
- Do not use outside knowledge.
- Do not reveal individual email addresses or phone numbers.
- Prefer shared HR or company contacts when present.
- Treat date-sensitive enrollment and deadline questions carefully.
- Return a strict JSON response schema.
- Classify each question into a known HR topic category.
- Ask a user-style follow-up question.

## Development Notes

- Keep API keys and connection strings out of the repository.
- Do not commit generated Chroma databases, PageIndex manifests, MP3 audio files, or chat logs unless they are intentionally sanitized fixtures.
- The highest-value tests would cover JSON parsing, retrieval selection, metadata normalization, Azure Table payload construction, and frontend API response handling.
