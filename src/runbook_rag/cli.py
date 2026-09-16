"""Command line entry point.

    runbook-rag fetch                 sparse-clone the documentation corpus
    runbook-rag index                 chunk, embed and load every strategy
    runbook-rag search "query"        one query against a chosen pipeline
    runbook-rag eval                  score the golden set, all configurations
    runbook-rag stats                 what is currently indexed
"""

from __future__ import annotations

import argparse
import sys
import time

from .chunking import STRATEGIES, chunk_documents
from .config import RAW_DIR
from .embedding import embed_passages, dimension
from .evaluate import comparison_table, run_eval, save_run, worst_queries
from .golden import load_golden
from .hybrid import BM25Retriever, HybridRetriever
from .ingest import fetch_source, load_documents, load_saved, save_documents
from .metrics import aggregate
from .rerank import RerankingRetriever, get_model as get_reranker
from .retrieve import DenseRetriever
from .store import load
from .store_pg import PgVectorStore
from .store_qdrant import QdrantStore

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def _store(name: str):
    return QdrantStore() if name == "qdrant" else PgVectorStore()


def _retriever(kind: str, store, strategy: str, depth: int):
    base = {
        "dense": lambda: DenseRetriever(store, strategy),
        "bm25": lambda: BM25Retriever(store, strategy),
        "hybrid": lambda: HybridRetriever(store, strategy),
    }
    if kind.startswith("rerank"):
        return RerankingRetriever(HybridRetriever(store, strategy), depth)
    return base[kind]()


def cmd_fetch(args) -> int:
    target = RAW_DIR / "source"
    print(f"fetching kubernetes/website into {target}")
    fetch_source(target)
    docs = load_documents(target)
    out = save_documents(docs)
    print(f"{len(docs)} documents -> {out}")
    return 0


def cmd_index(args) -> int:
    docs = load_saved()
    store = _store(args.store)
    strategies = [args.strategy] if args.strategy else list(STRATEGIES)
    print(f"{len(docs)} documents into {store.name}\n")

    for strategy in strategies:
        chunks = chunk_documents(docs, strategy)
        t = time.time()
        vectors = embed_passages([c.content for c in chunks])
        embedded = time.time() - t
        t = time.time()
        written = load(store, chunks, vectors, strategy)
        print(f"  {strategy:10} {written:>5} chunks  "
              f"{DIM}embed {embedded:.0f}s, index {time.time() - t:.0f}s{RESET}")
    print(f"\ntotal indexed: {store.count()}")
    return 0


def cmd_search(args) -> int:
    store = _store(args.store)
    retriever = _retriever(args.retriever, store, args.strategy, args.depth)
    t = time.time()
    hits = retriever.search(args.query, k=args.k)
    elapsed = time.time() - t

    print(f"{BOLD}{args.query}{RESET}")
    print(f"{DIM}{retriever.name}, {elapsed * 1000:.0f}ms{RESET}\n")
    for rank, hit in enumerate(hits, start=1):
        where = hit.doc_title + (f" > {hit.section}" if hit.section else "")
        print(f"{rank:>2}. {hit.score:6.3f}  {BOLD}{where}{RESET}")
        print(f"     {DIM}{hit.url}{RESET}")
        snippet = " ".join(hit.content.split())[:160]
        print(f"     {snippet}...\n")
    return 0


def cmd_eval(args) -> int:
    store = _store(args.store)
    queries = load_golden()
    print(f"{BOLD}{len(queries)} queries{RESET} against {store.name}\n")

    configurations = [
        ("dense / fixed", lambda: DenseRetriever(store, "fixed")),
        ("dense / header", lambda: DenseRetriever(store, "header")),
        ("bm25 / header", lambda: BM25Retriever(store, "header")),
        ("hybrid / header", lambda: HybridRetriever(store, "header")),
        ("hybrid + rerank", lambda: RerankingRetriever(HybridRetriever(store, "header"), args.depth)),
    ]
    if args.quick:
        configurations = configurations[:4]
    else:
        get_reranker()  # load once so the timing below is steady-state

    runs, last = {}, None
    for label, make in configurations:
        retriever = make()
        t = time.time()
        results = run_eval(retriever, queries)
        runs[label] = aggregate(results) | {
            "_s_per_query": round((time.time() - t) / len(queries), 3)
        }
        save_run(retriever.name, results, {"store": args.store})
        last = results
        print(f"  {label:22} recall@5 {runs[label]['recall_at_5']:.3f}  "
              f"{DIM}{runs[label]['_s_per_query']:.3f}s/query{RESET}")

    print(f"\n{comparison_table(runs)}\n")
    print(f"{BOLD}worst queries in the final configuration{RESET}")
    for result in worst_queries(last, n=5):
        query = next(q for q in queries if q.id == result.query_id)
        print(f"  {result.query_id}  mrr={result.mrr:.2f}  {query.text}")
    return 0


def cmd_stats(args) -> int:
    store = _store(args.store)
    print(f"store: {store.name}")
    for strategy in STRATEGIES:
        print(f"  {strategy:10} {store.count(strategy):>6} chunks")
    print(f"  {'total':10} {store.count():>6} chunks")
    print(f"embedding dimension: {dimension()}")
    print(f"golden queries: {len(load_golden())}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="runbook-rag", description=__doc__)
    parser.add_argument("--store", default="pgvector", choices=["pgvector", "qdrant"])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the corpus").set_defaults(func=cmd_fetch)

    p_index = sub.add_parser("index", help="chunk, embed and load")
    p_index.add_argument("--strategy", choices=sorted(STRATEGIES))
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="run one query")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--retriever", default="hybrid",
                          choices=["dense", "bm25", "hybrid", "rerank"])
    p_search.add_argument("--strategy", default="header", choices=sorted(STRATEGIES))
    p_search.add_argument("--depth", type=int, default=30)
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="score the golden set")
    p_eval.add_argument("--depth", type=int, default=100,
                        help="reranker candidate depth")
    p_eval.add_argument("--quick", action="store_true",
                        help="skip the reranked configuration")
    p_eval.set_defaults(func=cmd_eval)

    sub.add_parser("stats", help="what is indexed").set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
