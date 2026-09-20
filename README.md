# Autonomous arXiv Paper Digest & QA Agent

**Live demo:** _(add your deployed Streamlit Community Cloud link here)_

An agent that takes a research **topic** or a specific **arXiv id/URL**, retrieves
the paper via the official arXiv API, downloads and parses the PDF, builds a
local (TF-IDF) vector index over it, produces a structured **executive
briefing**, and then answers **grounded follow-up questions** about the paper
in an interactive CLI loop.

Built for the "Autonomous arXiv Paper Digest & QA Agent" take-home assessment.
No paid API key is required to run this — pick any one free-tier/local LLM
backend (Groq, Gemini, or Ollama) from `.env.example`.

---

## 1. Architecture — explicit state graph

The pipeline is implemented as an **explicit stateful graph** (`src/graph.py`
— ~70 lines, no external orchestration framework) of nodes that read/write a
single shared `AgentState` dataclass (`src/state.py`). Using a hand-rolled
graph instead of e.g. LangGraph was a deliberate choice for a project this
size — see [Design Decisions](#6-design-decisions--tradeoffs).

```
 query_understanding
         |
 arxiv_retrieval  ----(zero candidates)---------> END  ("no matching papers")
         |
   (>=1 candidate)
         v
 selection_ranking   (auto-picks top match; full ranked list is printed)
         |
   fetch_parse  ------(download/parse unrecoverable)--> END ("couldn't fetch")
         |
   (parsed_ok OR parsed_degraded — degraded just means
    "fall back to the arXiv abstract", not "abort")
         v
   chunk_embed  ------(no text at all to index)-------> END ("nothing to index")
         |
         v
    summarize  ---------------------------------------> briefing ready (END)


      ┌─────────────────────────────────────────────────────┐
      │  interactive QA loop (CLI REPL, NOT a graph node)    │
      │  reuses state.collection_name's persisted TF-IDF     │
      │  index + state.briefing; grows state.qa_history      │
      └─────────────────────────────────────────────────────┘
```

### Nodes (`src/nodes/*.py`)

| Node | Responsibility | Key state written |
|---|---|---|
| `query_understanding` | regex-detect an arXiv id/URL vs. free-text topic | `mode` |
| `arxiv_retrieval` | call the arXiv Atom API (`src/arxiv_client.py`) | `candidates`, `status` |
| `selection_ranking` | rank candidates by lexical overlap + recency, pick top-1 | `selected_paper` |
| `fetch_parse` | download PDF, extract text/sections (`src/pdf_parser.py`) | `pdf_path`, `parsed` |
| `chunk_embed` | chunk text, build local TF-IDF vector store | `chunks`, `collection_name` |
| `summarize` | retrieve grounding context, call the LLM, parse structured JSON | `briefing` |
| `qa` (not a graph node — called per REPL turn) | retrieve top-k chunks, refuse or answer | `qa_history` |

### State shape (`src/state.py`)

`AgentState` carries: `query`, `mode`, `candidates`, `selected_paper`,
`pdf_path`, `parsed` (`ParsedPaper`: full text + heuristically split
sections + parse warnings), `chunks`, `collection_name`, `briefing`
(`Briefing` dataclass matching the required output schema), `qa_history`
(list of `(question, answer)` pairs), plus `status`/`errors`/`trace` for
control flow and debugging (`--verbose` prints `trace` live).

### State persistence between summarize and the QA stage

Three layers, each answering "what if the process restarts":

1. **In-process**: the live `AgentState` object is held in memory for the
   duration of one CLI run — the fastest path, used while the REPL is open.
2. **Vector store**: `TfidfStore` (`src/vectorstore/tfidf_store.py`) persists
   the fitted vectorizer + TF-IDF matrix + chunk text to a pickle file per
   paper under `data/vectorstore/<arxiv_id>.pkl`, so re-embedding never has
   to happen twice for the same paper.
3. **Session file**: after a successful briefing, `save_session()`
   (`src/pipeline.py`) writes `sessions/<arxiv_id>.json` with the query,
   collection name, briefing, and QA history so far. `--resume <arxiv_id>`
   reloads that JSON plus the on-disk vector store and drops straight into
   the QA REPL — no arXiv/LLM calls needed to keep asking questions about a
   paper you've already processed.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching key. Pick ONE:
#   groq    -> free key at https://console.groq.com
#   gemini  -> free key at https://aistudio.google.com
#   ollama  -> `ollama pull llama3.1` locally, no key needed
```

`.env` is loaded automatically by both the CLI and the UI (via
`python-dotenv`) — no manual `export`/`source` step needed, and none of
that is shell-dependent (bash/zsh/fish all just work).

No vector DB service to stand up — the TF-IDF store is a local pickle file.
No paid dependency is required anywhere in `requirements.txt`.

## 3. Run

```bash
# Topic search
python -m src.cli --query "recent work on KV-cache compression for LLMs"

# Specific paper
python -m src.cli --query 2401.12345
python -m src.cli --query https://arxiv.org/abs/2401.12345

# Resume a previous session (no network calls needed except the QA answers)
python -m src.cli --resume 2401.12345

# Non-interactive (for scripting/grading)
python -m src.cli --query "..." --no-repl --json
```

### Example run

*(Illustrative — this sandbox's network is locked down to package
registries only, so the transcript below is a representative run rather
than a captured one; unit tests in `tests/` exercise the real code paths
against mocked arXiv/LLM responses. See `examples/sample_run.md` for the
full annotated version.)*

```
$ python -m src.cli --query "KV-cache compression for LLMs"
Running pipeline for: 'KV-cache compression for LLMs'

==============================================================================
KV-Cache Compression Survey for LLMs
B. Cee
arXiv:2401.99999  (2024-03-01)   http://arxiv.org/abs/2401.99999v1
==============================================================================

WHY IT MATTERS
This paper surveys techniques for shrinking the key-value cache used during
transformer inference, which is the main memory bottleneck for serving long
contexts, and organizes existing methods by where in the pipeline they act.

PROBLEM STATEMENT
KV-cache memory grows linearly with sequence length and batch size, limiting
achievable context length and batch throughput on fixed hardware.

METHOD
  - Categorizes approaches into quantization, token eviction, and low-rank
    projection of cached keys/values
  - Benchmarks each category on a shared long-context evaluation suite

KEY RESULTS
  - Quantization to 4-bit recovers >95% of full-precision accuracy at ~4x
    memory reduction on the surveyed benchmarks
  - Token-eviction methods trade a larger accuracy drop for the largest
    memory savings at very long contexts

LIMITATIONS
  - Survey coverage is limited to methods published before the cutoff date
  - Benchmarks are text-only; multimodal KV-cache behavior is not covered

SUGGESTED FOLLOW-UP QUESTIONS
  - Which quantization scheme has the best accuracy/memory tradeoff?
  - How do these methods interact with speculative decoding?
==============================================================================

Ask questions about this paper (blank line or 'exit' to quit).

> which quantization scheme performs best?
The survey reports 4-bit quantization recovering over 95% of full-precision
accuracy at roughly 4x memory reduction (from: method, results).

> does it cover video models?
I couldn't find anything in the retrieved text of this paper that answers
that -- it may not be covered, or may be phrased very differently than the
question.

> exit
```

## 4. Optional UI

`app.py` ("Paperroom") is a Streamlit workspace over the same
`src/pipeline.py` and `src/nodes/qa.py` used by the CLI — no changes to the
agent itself, purely a presentation layer.

```bash
pip install -r requirements.txt   # now includes streamlit
streamlit run app.py
```

It opens in your browser at `http://localhost:8501` with:
- a query box (topic / arXiv id / URL) and a "Research" button that streams
  live progress through the same graph nodes the CLI's `--verbose` flag shows
- an **Overview** tab: arXiv category chips, a "why it matters" and problem
  card, and Method/Key results/Limitations rendered as color-accented cards
  (blue/green/amber) rather than plain bullets, plus a "try asking" shortcut
  straight into Chat
- a **Chat** tab: grounded RAG conversation with clickable suggested
  questions (including a guaranteed "summarize this paper" chip) and an
  empty-state placeholder before the first message
- a **Source** tab: the parsed section text the RAG index was actually built
  from, for sanity-checking what the agent could see
- a sidebar with dark-mode toggle, which LLM provider/key is active, and a
  list of recent papers to reopen (same `sessions/*.json` files the CLI
  writes) without re-running retrieval/parsing/embedding

The Chat tab's grounding has two paths, matching the fix described in
[Design Decisions](#6-design-decisions--tradeoffs): broad questions
("summarize this", "what's it about") answer from the already-generated
briefing; specific questions go through normal chunk retrieval, with the
similarity-threshold refusal as a guardrail against hallucination.

> Note: the assessment brief lists "a frontend/UI beyond a basic CLI" as
> **out of scope** for grading. `app.py` is included as a convenience for
> your own use on top of the graded CLI deliverable, not a replacement for
> it — keep the CLI as what you point the reviewers at.

## 5. Testing

```bash
python -m pytest tests/ -v
```

23 tests, fully offline (arXiv responses, PDF files, and LLM calls are all
mocked/synthesized — no network needed to run the suite). Coverage
includes the two failure cases called out in §5 of the brief:

- **Zero / many candidates**: `test_zero_candidates_routes_to_end_without_crashing`,
  `test_many_candidates_picks_best_lexical_match`, `test_nonexistent_paper_id_routes_to_end`
- **PDF fails to parse cleanly**: `test_parse_blank_scanned_pdf_falls_back`,
  `test_parse_corrupted_file_path`, `test_pdf_download_failure_falls_back_to_abstract_and_still_produces_briefing`
- **Grounded QA / refusal to hallucinate**: `test_qa_grounded_answer_uses_llm`,
  `test_qa_refuses_when_similarity_too_low`
- **Broad/meta questions answer from the briefing, not retrieval**:
  `test_qa_meta_summary_question_uses_briefing_not_retrieval`,
  `test_qa_meta_question_detection_variants`

## 6. Design Decisions & Tradeoffs

**Custom state graph instead of LangGraph/CrewAI.** The graph here is a DAG
with one real branch point (candidate count) and two early-exit conditions.
A framework would add a dependency + its own state-typing conventions
without buying much at this scale; ~70 lines in `src/graph.py` keep every
edge visible in one place. With more time / a more complex graph (e.g.
parallel multi-paper digestion), LangGraph's checkpointing and parallel
fan-out would start paying for itself.

**TF-IDF instead of a neural embedding model for the vector store.**
`sklearn.TfidfVectorizer` + brute-force cosine similarity needs zero
downloaded model weights, is fully deterministic (important for the test
suite), and is exact rather than approximate at this scale (tens to low
hundreds of chunks per paper). It's weaker than a sentence-transformer on
paraphrased questions, but stronger on exact-terminology lookups typical of
technical papers ("LoRA rank", "KV-cache"). The swap is isolated to
`src/vectorstore/tfidf_store.py` — nothing else in the graph knows how
similarity is computed, so plugging in `sentence-transformers` + FAISS/Chroma
later is a localized change, not a redesign.

**Auto-select the top-ranked candidate instead of pausing for user input
mid-graph.** The brief explicitly allows CLI-only interaction and marks a
UI out of scope. Rather than block graph execution on stdin (awkward inside
a "run to completion" state machine), `selection_ranking` ranks all
candidates by lexical-overlap + recency and takes the top one, while the
full ranked list stays available on `state.candidates` for inspection. What
I'd do next: a `--pick <n>` CLI flag that replays selection against a
cached candidate list without re-hitting the arXiv API.

**Grounding / anti-hallucination strategy.** Two layers: (1) a hard
similarity-threshold gate — if the best-matching chunk's TF-IDF cosine
similarity is below `GROUNDING_THRESHOLD`, the QA node returns a canned
"not covered" response and never calls the LLM; (2) when the gate passes,
the LLM's system prompt still instructs it to answer only from the given
excerpts and say so if they're insufficient, and to cite which section it
used. Layer 1 is deterministic and free; layer 2 is a soft guardrail on top
of it — belt and suspenders rather than relying on either alone.

**Broad/meta questions bypass retrieval entirely.** Questions like
"summarize this" or "what's it about" share almost no vocabulary with any
single passage, so TF-IDF systematically under-scores them and they used
to trip the similarity gate and get refused even though the paper
obviously has an answer. `src/nodes/qa.py` now regex-detects this question
shape and answers from `state.briefing` instead — still grounded (the
briefing itself came from retrieved context during `summarize`), just via
a context source suited to a broad question instead of forcing a
narrow-retrieval tool to answer a wide-angle one.

**Section splitting is a heuristic, not a real layout parser.** Flattened
PDF text has no reliable structure markers once headers/body are on equal
footing, so `pdf_parser._split_sections` matches short lines against a
known list of section names (Abstract, Method, Results, ...). This is good
enough for chunking and for feeding `summarize`/`qa` section-tagged
excerpts, but will misfire on papers with unconventional section names or
two-column layouts that interleave oddly during text extraction. A more
robust version would use PyMuPDF's font-size/bbox metadata to detect
headers structurally instead of by string matching.

**Known limitations / what I'd do with more time:**
- No OCR fallback for genuinely scanned PDFs (out of scope per the brief's
  constraints list, but noted as a real gap for older papers).
- Ranking is lexical-overlap based, not learned/semantic — a topic phrased
  very differently from the paper's own vocabulary can under-rank a truly
  relevant result.
- Single-paper QA only; no cross-paper synthesis even though `candidates`
  holds the full ranked list.
- `GROUNDING_THRESHOLD` is a hand-picked constant, not calibrated against a
  labeled QA set.
- Meta-question detection (`_is_meta_summary_question`) is a fixed regex
  list, not a learned classifier — an unusual phrasing of "summarize this"
  could still slip through to the retrieval path and get refused.

## 7. Out of scope (per the brief)

No frontend beyond the CLI, no auth/deployment/production infra, no
non-arXiv sources, no fine-tuning.

## 8. Project layout

```
app.py                  optional Streamlit UI ("Paperroom" — layer over src/pipeline.py + src/nodes/qa.py)
.streamlit/config.toml  Streamlit theme for app.py
src/
  state.py              AgentState / PaperMeta / ParsedPaper / Chunk / Briefing
  graph.py               generic node/edge state-graph engine
  pipeline.py             wires nodes into the graph + session save/load
  arxiv_client.py          arXiv Atom API client
  pdf_parser.py            PDF download + text/section extraction
  chunking.py               paragraph-aware chunker with overlap
  cli.py                     CLI entry point + REPL
  nodes/                     one module per graph node (qa.py also handles meta-question routing)
  llm/                        pluggable LLM providers (groq/gemini/ollama/stub)
  vectorstore/                  local TF-IDF vector store
tests/                    23 offline unit tests (mocked network/LLM)
sessions/                 saved QA sessions (git-ignored)
data/                     downloaded PDFs + persisted vector stores (git-ignored)
```
