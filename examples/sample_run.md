# Annotated sample run

This mirrors the "Example run" in the README but includes the `--json`
briefing output shape and a second QA exchange, to fully satisfy the
deliverable ("input -> briefing output -> 2-3 sample QA exchanges").

> Note: produced against mocked arXiv/LLM responses (same fixtures the test
> suite uses) since this environment cannot reach arxiv.org or any LLM API.
> Running the commands below for real just needs `.env` filled in and
> internet access — nothing else changes.

## 1. Input

```
python -m src.cli --query "KV-cache compression for LLMs"
```

## 2. Briefing output (`--json` form)

```json
{
  "title": "KV-Cache Compression Survey for LLMs",
  "authors": ["B. Cee"],
  "arxiv_id": "2401.99999",
  "published": "2024-03-01T00:00:00Z",
  "link": "http://arxiv.org/abs/2401.99999v1",
  "summary": "This paper surveys techniques for shrinking the key-value cache used during transformer inference, which is the main memory bottleneck for serving long contexts, and organizes existing methods by where in the pipeline they act.",
  "problem_statement": "KV-cache memory grows linearly with sequence length and batch size, limiting achievable context length and batch throughput on fixed hardware.",
  "method": [
    "Categorizes approaches into quantization, token eviction, and low-rank projection of cached keys/values",
    "Benchmarks each category on a shared long-context evaluation suite"
  ],
  "key_results": [
    "Quantization to 4-bit recovers >95% of full-precision accuracy at ~4x memory reduction on the surveyed benchmarks",
    "Token-eviction methods trade a larger accuracy drop for the largest memory savings at very long contexts"
  ],
  "limitations": [
    "Survey coverage is limited to methods published before the cutoff date",
    "Benchmarks are text-only; multimodal KV-cache behavior is not covered"
  ],
  "suggested_questions": [
    "Which quantization scheme has the best accuracy/memory tradeoff?",
    "How do these methods interact with speculative decoding?"
  ]
}
```

## 3. Sample QA exchanges

**Q1 — answerable from retrieved context**
```
> which quantization scheme performs best?
The survey reports 4-bit quantization recovering over 95% of full-precision
accuracy at roughly 4x memory reduction (from: method, results).
```

**Q2 — out of scope for the paper (grounding refusal, no LLM call made)**
```
> does it cover video models?
I couldn't find anything in the retrieved text of this paper that answers
that -- it may not be covered, or may be phrased very differently than the
question. Try rephrasing, or note this may be beyond what was parsed from
the PDF.
```

**Q3 — answerable, cites a different section**
```
> what's the main limitation the authors mention?
The survey notes its coverage is limited to methods published before its
cutoff date, and that benchmarks are text-only, so multimodal KV-cache
behavior isn't addressed (from: limitations).
```

## 4. Resuming later

```
$ python -m src.cli --resume 2401.99999
Ask questions about this paper (blank line or 'exit' to quit).

> <continue asking, no arXiv/LLM calls needed except for the answers themselves>
```
