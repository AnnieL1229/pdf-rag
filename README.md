# PDF RAG

Upload PDFs, ask questions in plain language, and get answers grounded in those documents—with **citations** (file and page). Hybrid search (vectors + keywords) and indexes live locally under **`data/`**; **Mistral** powers routing, coverage checks, and answer text when **`MISTRAL_API_KEY`** is set. Use the **FastAPI** server, the **Streamlit** UI, or any HTTP client.

Section 1 diagrams the pipeline; sections 2–6 explain each stage, then API, UI, how to run, limitations, and repo layout.

---

## 1. System architecture

FastAPI holds **singletons** on `app.state`: a `KnowledgeBase` (chunks + indexes), `QueryProcessor`, `AnswerGenerator`, and `AmbiguityChecker` (`app/main.py`). Each `/query` runs the same staged pipeline.

**End-to-end flow**

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

---

## 2. Data ingestion

Uploads go to **`POST /ingest`** with multipart field **`files`**. Non-PDFs are skipped with warnings (`routes_ingest.py`). For each PDF, **PyMuPDF** reads text page by page (`pdf_parser.py`); whitespace is normalized in **`text.py`**. There is **no OCR**, so scans may produce little usable text.

**`TextChunker`** (`chunker.py`, limits in `config.py`) splits into overlapping segments—by default about **900** characters with **150** overlap—preferring paragraph boundaries, then sentences, then fixed windows. Each chunk stores **`chunk_id`**, **`filename`**, **`page_number`**, and **`text`** in **`KnowledgeBase`** (`storage.py`). Ingestion **appends** to `data/chunks.json` and updates the FAISS index; BM25 is rebuilt over the **entire** corpus each time. Before LLM steps, **`attach_neighbor_context`** can stitch neighboring chunks from the **same file** so split tables or definitions still appear in one prompt.

---

## 3. Query processing

**`QueryProcessor`** (`query_processor.py`) decides whether the message needs document retrieval. When **`MISTRAL_API_KEY`** is set, Mistral returns a small **JSON** object: **`route`** (greeting, thanks, help, retrieval, refusal), **`needs_retrieval`**, **`answer_format`**, and optionally **`rewritten_query`**—a compact search string that keeps proper nouns and quantifiers where the prompt asks for that.

If the key is missing, the call fails, or JSON is invalid, the code falls back to simple patterns for hello/thanks/help, **regex-based refusal** for sensitive phrasing (PII, medical, legal), and otherwise treats the input as a document question and uses the **original** wording for search.

---

## 4. Retrieval strategy

**Semantic** search uses **`sentence-transformers/all-MiniLM-L6-v2`**: chunks and the query are embedded, L2-normalized, and matched with **FAISS `IndexFlatIP`** (inner product ≈ cosine similarity). **BM25** runs in parallel on the same chunk texts (`text.py`). Together they cover paraphrases and exact tokens better than either channel alone.

**`fuse_results`** (`retriever.py`) joins the two hit lists on **`chunk_id`**, rescales scores into a common range, adds a small query–chunk overlap boost, and penalizes near-duplicate bodies. Defaults pull **eight** semantic and **eight** keyword candidates, then keep **five** after fusion. There is no cross-encoder reranker. **`evidence_is_strong`** rejects weak fused scores before spending tokens on coverage or answering.

---

## 5. Coverage, answerability, and ambiguity

Some questions need evidence for several parts at once (comparisons, multi-entity lists). Strong retrieval for **one** part can still invite confident-sounding answers that skip the rest. **`AmbiguityChecker.detect`** (`ambiguity.py`) runs **after** the evidence gate, on chunks already expanded with neighbor text. Mistral returns JSON: **`coverage_sufficient`**, **`needs_clarification`**, **`missing_components`**, **`reason`**, and **`clarification_question`** when useful.

**`routes_query.py`** maps that to HTTP behavior: if **`needs_clarification`**, the answer field is empty and the client gets a clarification string plus sources; if coverage is insufficient but not a clarification case, **`AnswerGenerator`** may run in **partial** mode with stricter instructions; if coverage looks good, the normal answer path runs. When the validator returns nothing usable, the handler prefers a **conservative** outcome over pretending validation passed. The first sentence of **`clarification_question`** is kept short for stable UI.

---

## 6. Generation

**`AnswerGenerator`** (`generator.py`) builds a single prompt where each passage is labeled with **filename**, **page**, and **chunk id**, with body text often widened by neighbor stitching. Prompts instruct Mistral to stay in-context, admit unknowns, and follow **`answer_format`** (bullets, a small table when appropriate, a terse answer, or a short paragraph).

The API response always includes **`SourceChunk`** rows for the fused top hits (with scores), so citations remain even if the prose omits a page. **`filter_by_evidence`** drops answer sentences whose tokens barely overlap the retrieved passages—a fast lexical guardrail, not a full entailment model.

---

## 7. API endpoints

| Method | Path | Body | Notes |
|--------|------|------|--------|
| `POST` | `/ingest` | `multipart/form-data`, field `files` | Accepts one or more PDFs; response includes counts, filenames, and warnings. |
| `POST` | `/query` | JSON `{"question": "..."}` | Response shape is **`QueryResponse`** in `app/models/schemas.py`. |
| `GET` | `/` | — | Simple JSON liveness check. |

---

## 8. UI

The Streamlit client in **`ui/streamlit_app.py`** lets you point at an API base URL (it defaults to **`http://localhost:8000`**), upload files to **`/ingest`**, and submit questions to **`/query`**. The layout shows the main answer when there is one, surfaces clarification as a warning when the API asks for it, prints a short validation line when something was off about evidence or coverage, shows a compact metadata row, and lists each retrieved chunk in an expander with filename, page, fused score, and text.

---

## 9. How to run

```bash
cd "/path/to/StackAI - RAG"
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add MISTRAL_API_KEY to .env
```

Start the API:

```bash
uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

In another terminal, start the UI:

```bash
streamlit run ui/streamlit_app.py
# http://127.0.0.1:8501
```

Optional smoke checks:

```bash
pytest
curl -s -X POST "http://127.0.0.1:8000/ingest" -F "files=@./sample.pdf"
curl -s -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main terms in section 3?"}'
```

---

## 10. Limitations and future work

The stack assumes PDFs already contain extractable text; there is **no OCR**, so scanned or image-only pages often yield little or nothing. Documents are added by **append**: there is no API to delete, replace, or version a single upload, and clearing the corpus means removing files under **`data/`** yourself. BM25 is rebuilt over the full corpus on each ingest, which is acceptable for small workloads but not tuned for heavy concurrent writes. This repo also does not include a labeled benchmark or production-style monitoring.

If you extended the project, natural directions would be a stronger **reranker** (for example a cross-encoder) over more candidates, chunking that respects **tables or multi-column layout**, **multilingual** or larger embedding models, **separate indexes per tenant**, and explicit **lifecycle** endpoints to remove or re-ingest documents with traceable metadata.

---

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
.env.example
```
