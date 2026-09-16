# Runbook RAG

A retrieval pipeline over real operational documentation, built to be measured
rather than demonstrated. Chunking, embedding, lexical search, fusion and
reranking are each swappable, and every one of them is scored against a hand
labelled query set.

Retrieval quality improves by **+21% recall@5** over a dense baseline. The
interesting part is the rest of the table: what did not work, and what the
improvement cost.

Everything runs locally. No API key, no per-token cost, and identical numbers on
every run.

## Results

381 Kubernetes documentation pages, 70 labelled queries, macro-averaged.

| configuration | recall@5 | recall@10 | MRR | nDCG@5 | hit@5 | s/query |
| --- | --- | --- | --- | --- | --- | --- |
| dense, header chunks (baseline) | 0.526 | 0.721 | 0.575 | 0.481 | 0.757 | 0.006 |
| BM25 only | 0.486 | 0.591 | 0.585 | 0.458 | 0.743 | 0.004 |
| hybrid (dense + BM25, RRF) | 0.564 | 0.679 | 0.643 | 0.536 | 0.829 | 0.013 |
| hybrid + cross-encoder rerank | **0.638** | 0.707 | **0.659** | **0.573** | **0.871** | 1.203 |
| | **+21%** | −2% | **+15%** | **+19%** | **+15%** | **×200** |

`hit@5` is the number to look at if the result feeds a language model: it went
from 0.757 to 0.871, so the share of queries where nothing useful reached the
context window fell from roughly one in four to one in eight.

Reproduce with `runbook-rag eval`.

### What did not work

**Header-aware chunking lost to fixed windows on the metric I expected it to
win.** Prefixing each chunk with its document title and section should help, and
it does — but only deeper in the list.

| chunking | chunks | recall@5 | recall@10 | MRR |
| --- | --- | --- | --- | --- |
| fixed windows | 2,875 | 0.545 | 0.669 | 0.604 |
| recursive (paragraph-aware) | 2,934 | 0.533 | 0.681 | 0.572 |
| header-aware | 4,549 | 0.526 | 0.721 | 0.575 |

Header chunks are smaller and more numerous, which surfaces more distinct
documents by rank 10 (+8%) while diluting the top five. I had written the
opposite into the code as a comment before measuring it; the comment is now
corrected rather than removed, because being wrong about that was the most
useful thing the eval did.

**BM25 alone is clearly worse than dense retrieval** — 11% lower recall@5. It
still improves the system by 7% when fused, because it fails on different
queries. Exact identifiers like `CrashLoopBackOff` are where embeddings are
weakest and lexical matching is strongest.

**Reranking plateaus at the size of the candidate pool, not at the depth you
configure.** Candidate depths of 100 and 150 score identically, because the
hybrid retriever only ever produces about a hundred distinct candidates. A
reranker cannot promote a document that was never retrieved — the retriever sets
the ceiling and the reranker only decides the order beneath it.

**The reranker costs 200x latency for that 21%.** 6ms to 1.2s per query on CPU.
Whether that is worth paying depends entirely on whether a human or a language
model is waiting, and that trade-off is the actual finding, not the +21%.

## The labelled query set

Seventy queries across 27 facets, 143 graded labels, in `eval/golden.yaml`.
This is the part that took the longest and it is the only reason any of the
numbers above mean anything.

Queries are written the way an engineer under pressure types them — a symptom,
an error string, a half-remembered flag — and deliberately **not** phrased as
the titles of the documents they should match:

```yaml
- id: Q-030
  query: traffic between two services started timing out with no errors on either side
  facet: network-policy
  relevant:
    concepts/services-networking/network-policies.md: 2
    tasks/administer-cluster/declare-network-policy.md: 2
```

Grades are `2` for a document that answers the question and `1` for genuine
supporting context, which is what nDCG needs to distinguish a good ranking from
a merely correct one.

Labels are written as repository paths, not document ids, so they can be
reviewed in a diff. They resolve to ids at load time, and a path that no longer
exists **raises** rather than scoring zero — upstream documentation moves, and a
moved page would otherwise quietly depress every metric with no indication why.
A test asserts the whole set resolves, and another asserts no query simply
restates the title of its own answer.

## Design notes

**Metrics are computed at document level, not chunk level.** Chunking is a
variable here, so a chunk-level metric would reward whichever strategy happened
to slice a relevant page into more pieces. Scoring the document a chunk belongs
to is what makes the three strategies comparable.

**Fusion is by reciprocal rank, not by weighted score.** Cosine similarity is
bounded and BM25 is not, so any weighted sum of raw scores is tuning a constant
with no meaning. RRF uses only positions, which both rankings agree on.

**Two vector stores, one interface.** pgvector is the default; Qdrant runs
embedded with no server. The parity suite asserts both behave identically
through the `VectorStore` interface, which is what proves the pipeline is not
quietly depending on one of them.

**The HNSW index is built after loading.** Building it on an empty table and
filling row by row is slower and produces a worse graph. `hnsw.ef_search` is
raised from its default of 40, which is low enough to lose neighbours at this
corpus size.

**BGE prefixes queries but not passages.** Getting this backwards costs recall
silently, so the two paths are separate functions rather than a flag.

## Running it

Requires Python 3.11+ and Postgres with pgvector.

```bash
brew install postgresql@17 pgvector && brew services start postgresql@17
createdb runbook_rag
# or: docker compose up -d   (then RUNBOOK_RAG_DSN=postgresql://runbook:runbook@localhost:5433/runbook_rag)

uv venv --python 3.12 && uv pip install -e ".[dev]"

runbook-rag fetch        # sparse-clone the corpus
runbook-rag index        # chunk, embed and load all three strategies
runbook-rag eval         # score the golden set, print the table above
runbook-rag search "pods evicted because the node ran out of memory"
runbook-rag stats
```

`--store qdrant` switches backend on any command. `pytest` runs 120 tests;
`-m "not slow"` skips the ones that load a model.

First run downloads about 500MB of model weights and takes a few minutes to
embed. After that, indexing is cached and an eval sweep takes under two minutes.

## Layout

| path | what it holds |
| --- | --- |
| `eval/golden.yaml` | the labelled query set |
| `src/runbook_rag/chunking.py` | three chunking strategies |
| `src/runbook_rag/embedding.py` | local BGE embeddings, disk-cached |
| `src/runbook_rag/store_pg.py` | pgvector backend |
| `src/runbook_rag/store_qdrant.py` | Qdrant backend, same interface |
| `src/runbook_rag/hybrid.py` | BM25 and reciprocal rank fusion |
| `src/runbook_rag/rerank.py` | cross-encoder reranking |
| `src/runbook_rag/metrics.py` | recall@k, MRR, nDCG@k |

## Corpus

Kubernetes documentation (`kubernetes/website`, CC BY 4.0), tasks and concepts
sections: 381 pages, 3.8M characters. Real published documentation rather than
synthetic text, because real docs repeat themselves, contradict older pages,
bury the answer mid-page and share vocabulary across unrelated topics. Those are
the conditions a retriever actually has to work under.

## Licence

MIT
