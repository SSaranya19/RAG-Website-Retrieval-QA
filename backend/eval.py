"""
Retrieval-only evaluation harness for a RAG pipeline.

- Computes per-query precision/recall/F1, AP@k, RR, MAP, MRR
- Deduplicates & normalizes URLs before matching
- CLI: python eval.py --testfile testset.json --topk 5 --out results.json --verbose

Place this file where it can import `backend.engine.engine`.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
from urllib.parse import urlparse, urlunparse

# Project import
from backend.engine import engine

# ----------------- Utilities -----------------

def normalize_url(url: str) -> str:
    """
    Basic normalization for URL matching:
    - Remove trailing slash
    - Lowercase scheme+netloc
    - Remove fragment
    - Keep path and query
    """
    if not url:
        return url
    try:
        p = urlparse(url.strip())
        scheme = (p.scheme or "http").lower()
        netloc = p.netloc.lower()
        # remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        path = p.path.rstrip("/")  # remove trailing slash
        new = urlunparse((scheme, netloc, path, "", p.query, ""))  # drop params & fragment
        return new
    except Exception:
        return url.strip().rstrip("/")

def _unique_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ----------------- Retrieval metric helpers -----------------

def precision_recall_f1(retrieved: set, relevant: set) -> Tuple[float, float, float]:
    if not retrieved:
        prec = 0.0
    else:
        prec = len(retrieved & relevant) / len(retrieved)
    if not relevant:
        rec = 0.0
    else:
        rec = len(retrieved & relevant) / len(relevant)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1

def average_precision_at_k(retrieved_list: List[str], relevant_set: set, k: int) -> float:
    """
    AP@k with deduplication of retrieved_list (preserve first occurrence).
    Returns value in [0,1].
    """
    if not relevant_set:
        return 0.0
    retrieved_uniq = _unique_preserve_order(retrieved_list)[:k]
    hits = 0
    sum_precisions = 0.0
    for i, doc in enumerate(retrieved_uniq, start=1):
        if doc in relevant_set:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    denom = min(len(relevant_set), k)
    return sum_precisions / denom

def reciprocal_rank(retrieved_list: List[str], relevant_set: set) -> float:
    retrieved_uniq = _unique_preserve_order(retrieved_list)
    for i, doc in enumerate(retrieved_uniq, start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0

# ----------------- Core evaluation -----------------

def evaluate_retrieval(testset: List[Dict], top_k: int = 5, verbose: bool = False) -> Dict:
    """
    testset: list of {"query": str, "relevant_urls": [str], ...}
    Returns aggregated retrieval metrics and per-query breakdown.
    """
    per_query = []
    sum_ap = 0.0
    sum_rr = 0.0

    micro_tp = 0
    micro_retrieved = 0
    micro_relevant = 0

    for item in testset:
        query = item.get("query", "").strip()
        raw_relevant = item.get("relevant_urls", []) or []
        # Normalize relevant URLs
        relevant_urls = {normalize_url(u) for u in raw_relevant if u}

        if not query:
            per_query.append({
                "query": query,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "rr": 0.0,
                "retrieved": [],
                "relevant": list(relevant_urls),
                "note": "empty query"
            })
            continue

        # Retrieve using engine
        try:
            hits = engine.retrieve(query, top_k=top_k) or []
        except Exception as e:
            hits = []
            if verbose:
                print(f"[evaluate_retrieval] engine.retrieve error for query '{query}': {e}")

        retrieved_list_raw = [h.get("metadata", {}).get("url") for h in hits if h.get("metadata")]
        # Normalize and deduplicate preserving first occurrence
        retrieved_list = _unique_preserve_order([normalize_url(u) for u in retrieved_list_raw if u])
        retrieved_set = set(retrieved_list)

        prec, rec, f1 = precision_recall_f1(retrieved_set, relevant_urls)
        ap = average_precision_at_k(retrieved_list, relevant_urls, top_k)
        rr = reciprocal_rank(retrieved_list, relevant_urls)

        per_query.append({
            "query": query,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "ap": ap,
            "rr": rr,
            "retrieved": retrieved_list,
            "relevant": list(relevant_urls)
        })

        sum_ap += ap
        sum_rr += rr

        micro_tp += len(retrieved_set & relevant_urls)
        micro_retrieved += len(retrieved_set)
        micro_relevant += len(relevant_urls)

        if verbose:
            print(f"Q: {query}")
            print(f"  retrieved: {retrieved_list}")
            print(f"  relevant : {list(relevant_urls)}")
            print(f"  P={prec:.3f} R={rec:.3f} F1={f1:.3f} AP={ap:.3f} RR={rr:.3f}\n")

    qcount = len(per_query) or 1
    macro_precision = sum(p["precision"] for p in per_query) / qcount
    macro_recall = sum(p["recall"] for p in per_query) / qcount
    macro_f1 = sum(p["f1"] for p in per_query) / qcount
    map_score = sum_ap / qcount
    mrr = sum_rr / qcount

    micro_precision = (micro_tp / micro_retrieved) if micro_retrieved else 0.0
    micro_recall = (micro_tp / micro_relevant) if micro_relevant else 0.0
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    return {
        "per_query": per_query,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "MAP": map_score,
        "MRR": mrr,
        "top_k": top_k
    }

# ----------------- CLI -----------------

def load_testset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_results(outpath: str, data: Dict):
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testfile", "-t", required=True, help="Path to testset JSON")
    parser.add_argument("--topk", "-k", default=5, type=int, help="top_k for retrieval")
    parser.add_argument("--out", "-o", default="eval_results.json", help="output file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    testset = load_testset(args.testfile)

    print("Running retrieval evaluation...")
    retrieval_results = evaluate_retrieval(testset, top_k=args.topk, verbose=args.verbose)

    results = {"retrieval": retrieval_results}
    save_results(args.out, results)
    print(f"Saved results to {args.out}")

    # Summary
    print("\n=== Evaluation Summary ===")
    print(f"Retrieval (MAP={retrieval_results['MAP']:.4f}, MRR={retrieval_results['MRR']:.4f})")
    print(f"Retrieval (macro P/R/F1) = {retrieval_results['macro_precision']:.4f} / {retrieval_results['macro_recall']:.4f} / {retrieval_results['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
