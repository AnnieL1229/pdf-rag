# PDF RAG

Upload PDFs, ask questions in plain language, and get answers grounded in those documents—with **citations** (file and page). Hybrid search (vectors + keywords) and indexes live locally under **`data/`**; **Mistral** powers routing, coverage checks, and answer text when **`MISTRAL_API_KEY`** is set. Use the **FastAPI** server, the **Streamlit** UI, or any HTTP client.

Section 1 diagrams the pipeline; sections 2–6 explain each stage, then API, UI, how to run, limitations, and repo layout.


## 1. System architecture

```text
User question
      |
      v
+------------------+
| Query processor  |  intent (route), needs_retrieval, answer_format, rewritten_query
| (Mistral + rules)|
+------------------+
      |
      +-- no retrieval --> fixed short answer (greeting / help / refusal)
      |
      v
+------------------+
| Knowledge base   |  semantic_top_k + keyword_top_k candidates
| hybrid search    |  --> fuse_results --> final_top_k chunks
+------------------+
      |
      v
+------------------+
| Evidence gate    |  heuristic: top fused scores strong enough?
+------------------+
      |
      v
+------------------+
| Neighbor context |  merge adjacent chunks (same file) for LLM context window
+------------------+
      |
      v
+------------------+
| Coverage check   |  Mistral JSON: coverage_sufficient, needs_clarification, ...
| (AmbiguityChecker)|
+------------------+
      |
      +-- needs clarification --> response: clarification question + sources
      +-- coverage insufficient   --> partial answer attempt (bounded) + sources
      +-- coverage OK             --> full generation + sources
      |
      v
+------------------+
| Answer generator |  Mistral: grounded prompt + format hints
| + sentence filter|  drop sentences with low lexical overlap vs retrieved text
+------------------+
      |
      v
Response (answer, metadata, citations / retrieved_chunks)
```


## 2. Data ingestion

**Endpoint:** `POST /ingest` (multipart field `files`).

- Non-PDF files are skipped with warnings.
- Text extraction uses PyMuPDF (**no OCR**).

**Chunking**

- ~900 characters per chunk, ~150 overlap (defaults in `config.py`; logic in `chunker.py`).
- **Strategy:** paragraph → sentence → fallback split.
- **Goal:** keep chunks readable while bounding size.

**Storage**

Each chunk stores:

- `chunk_id`
- `filename`
- `page_number`
- `text`

**Notes**

- New documents are **appended** to `data/chunks.json`.
- The **FAISS** index is updated incrementally as chunks are added.
- **BM25** is rebuilt over the **full** corpus after each ingest.
- Neighbor chunks from the same file can be **stitched** (`attach_neighbor_context`) before LLM calls.


## 3. Query processing

Handled by **`QueryProcessor`** (`query_processor.py`).

**With Mistral** (when `MISTRAL_API_KEY` is set), the model returns JSON:

- `route` — greeting, help, retrieval, refusal, etc.
- `needs_retrieval`
- `answer_format`
- `rewritten_query` — search-optimized wording

**Fallback** (no API / invalid JSON / failed call):

- Simple patterns for greeting / help.
- Regex-based refusal detection for sensitive topics.
- Otherwise treat the message as a document query and use the original text for retrieval.


## 4. Retrieval strategy

**Two parallel channels**

*Semantic search*

- `all-MiniLM-L6-v2` embeddings (`sentence-transformers`).
- FAISS `IndexFlatIP` (cosine-like on L2-normalized vectors).

*Keyword search*

- BM25 over chunk text (`text.py`).

**Why hybrid**

- Semantic → handles paraphrases.
- BM25 → preserves exact tokens (IDs, numbers, phrases).

**Fusion — `fuse_results`** (`retriever.py`)

- Merge on `chunk_id`.
- Normalize scores.
- Add lexical overlap boost.
- Penalize near-duplicate chunks.

**Default**

- Top **8** semantic + top **8** keyword → top **5** after fusion.

**Filtering**

- `evidence_is_strong` rejects weak matches early.
- No cross-encoder reranker (kept lightweight).


## 5. Coverage, answerability, and ambiguity

Some queries need multiple pieces of evidence (e.g. comparisons). Retrieval may only cover part of the request.

**`AmbiguityChecker.detect`** (`ambiguity.py`) runs after retrieval (on neighbor-expanded chunks) and returns:

- `coverage_sufficient`
- `needs_clarification`
- `missing_components`
- `reason`
- `clarification_question`

**Behavior**

- Clarification needed → return clarification + sources.
- Partial coverage → generate a **bounded partial** answer.
- Sufficient coverage → normal answer path.

If validation fails or returns invalid output, the system falls back to a **conservative** path.

Ambiguity detection and coverage validation happen in the **same** Mistral call.


## 6. Generation

Handled by **`AnswerGenerator`** (`generator.py`).

**Prompt construction**

- Each chunk includes: **filename**, **page**, **chunk id**.
- Neighbor chunks may be merged for context.

**Instructions to the LLM**

- Stay within provided context.
- Do not guess missing information.
- Follow `answer_format`.

**Output**

- Answer text.
- Structured citations (`SourceChunk` objects for top fused hits).

**Post-processing**

- `filter_by_evidence` removes sentences with low lexical overlap vs retrieved text.
- Acts as a lightweight guardrail, **not** full verification / entailment.


## 7. API endpoints

| Method | Path | Body | Notes |
|--------|------|------|--------|
| `POST` | `/ingest` | `multipart/form-data`, field `files` | Accepts one or more PDFs; response includes counts, filenames, and warnings. |
| `POST` | `/query` | JSON `{"question": "..."}` | Response shape is **`QueryResponse`** in `app/models/schemas.py`. |
| `GET` | `/` | — | Simple JSON liveness check. |


## 8. UI

The Streamlit client in **`ui/streamlit_app.py`** lets you point at an API base URL (it defaults to **`http://localhost:8000`**), upload files to **`/ingest`**, and submit questions to **`/query`**. The layout shows the main answer when there is one, surfaces clarification as a warning when the API asks for it, prints a short validation line when something was off about evidence or coverage, shows a compact metadata row, and lists each retrieved chunk in an expander with filename, page, fused score, and text.


## 9. How to run

**Prerequisites:** Python 3.10+ recommended, a **Mistral API key** (for `/query` answers; routing and generation call Mistral). First time you embed or ingest, **sentence-transformers** downloads the embedding model from the network (one-time, may take a minute).

**1. Clone and enter the repo** (replace the URL with yours):

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO
```

**2. Virtual environment and dependencies** (from the **repository root**, the folder that contains `app/` and `requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Environment file**

```bash
cp .env.example .env
# Edit .env: set MISTRAL_API_KEY=... (and optionally MISTRAL_CHAT_MODEL)
```

**4. Start the API** (still from repo root):

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` — you should see a small JSON message from `GET /`.

**5. Start the UI** (second terminal, same `venv` and repo root):

```bash
streamlit run ui/streamlit_app.py
```

Use the UI to upload at least one PDF, then ask a question—or call `POST /ingest` and `POST /query` yourself.

**Checks**

```bash
pytest
```

The repo includes **`pyproject.toml`** so `pytest` resolves the `app` package without setting `PYTHONPATH` manually.

**curl** (replace `./your.pdf` with a real PDF path on your machine; ingest before query):

```bash
curl -s -X POST "http://127.0.0.1:8000/ingest" -F "files=@./your.pdf"
curl -s -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main terms in section 3?"}'
```

## 10. Limitations

The stack assumes PDFs already contain extractable text; there is **no OCR**, so scanned or image-only pages often yield little or nothing. Documents are added by **append**: there is no API to delete, replace, or version a single upload, and clearing the corpus means removing files under **`data/`** yourself. BM25 is rebuilt over the full corpus on each ingest, which is acceptable for small workloads but not tuned for heavy concurrent writes. This repo also does not include a labeled benchmark or production-style monitoring.

## 11. Repository layout

Backend code lives under **`app/`**; **`ui/`** is the Streamlit client; **`tests/`** are pytest modules; **`data/`** holds generated index files (gitignored).

```text
app/
  main.py                 # FastAPI entry, routers, app.state singletons
  api/                    # routes_ingest.py, routes_query.py
  core/config.py
  models/schemas.py
  services/               # PDF → chunks → embeddings → KnowledgeBase, retrieval, Mistral calls
  utils/                  # text.py, mistral_chat.py
ui/streamlit_app.py
tests/                    # test_*.py
data/                     # chunks.json, faiss.index (after ingest)
requirements.txt
pyproject.toml            # pytest pythonpath so tests find app
.env.example
```
