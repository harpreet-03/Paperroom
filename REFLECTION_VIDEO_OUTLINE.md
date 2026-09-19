# 4-minute reflection video — suggested outline

The brief asks for a short (max 4 min) video on your approach. This is a
talking outline for you to record — not a deliverable I can produce for
you. Rough timing for ~4 minutes:

**0:00–0:45 — What you built**
- One sentence: an agent that turns a topic/paper id into a structured
  briefing and answers grounded follow-up questions.
- Show the state graph diagram from the README, name the 6 nodes quickly.

**0:45–1:30 — Why a custom state graph**
- Explain the DAG shape: linear pipeline + one branch point (candidate
  count) + two early-exit conditions for failure cases.
- Mention why you didn't reach for LangGraph here (size of the problem
  doesn't justify the dependency) — this is the kind of tradeoff reasoning
  the rubric is looking for.

**1:30–2:15 — Retrieval/grounding design**
- TF-IDF local vector store: why (no model download, deterministic,
  exact at this scale), and the acknowledged weakness (no paraphrase
  robustness).
- The two-layer anti-hallucination guardrail: similarity threshold gate +
  "answer only from excerpts" system prompt.

**2:15–3:00 — Failure handling**
- Walk through what happens on: zero arXiv results, a scanned/broken PDF,
  a many-candidate topic search. Point at the specific tests that cover
  each (`tests/test_graph_flow.py`, `tests/test_pdf_parser.py`).

**3:00–3:45 — Live demo**
- Run `python -m src.cli --query "<your topic>"`, show the briefing, ask
  2 questions live (one grounded, one out-of-scope to show the refusal).

**3:45–4:00 — What you'd do next**
- Pick 1-2 items from the README's "known limitations" list (e.g. semantic
  ranking instead of lexical overlap, or `--pick <n>` for candidate
  selection).

Recording tip: screen-record the terminal for the demo section and just
talk over your editor/README for the design sections — keeps it inside 4
minutes without needing slides.
