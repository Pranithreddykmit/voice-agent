"""
Standard Retrieval Evaluation Metrics.
Implements Hit@K, Recall@K, Precision@K, MRR@K, and nDCG@K with strict [0, 1] normalization.
"""

import math
from typing import List, Set, Union


def hit_at_k(retrieved_items: List[Union[int, str]], ground_truth: List[Union[int, str]], k: int) -> float:
    """
    Hit@K: Returns 1.0 if at least one ground-truth item appears in top-K, else 0.0.
    """
    if not ground_truth:
        return 1.0 if not retrieved_items else 0.0
    top_k_set = set(retrieved_items[:k])
    gt_set = set(ground_truth)
    return 1.0 if bool(top_k_set & gt_set) else 0.0


def recall_at_k(retrieved_items: List[Union[int, str]], ground_truth: List[Union[int, str]], k: int) -> float:
    """
    Recall@K: Proportion of unique relevant items retrieved in top-K.
    """
    if not ground_truth:
        return 1.0
    top_k_set = set(retrieved_items[:k])
    gt_set = set(ground_truth)
    hits = len(top_k_set & gt_set)
    return float(hits) / float(len(gt_set))


def precision_at_k(retrieved_items: List[Union[int, str]], ground_truth: List[Union[int, str]], k: int) -> float:
    """
    Precision@K: Proportion of top-K retrieved items that are relevant.
    """
    if k <= 0:
        return 0.0
    if not ground_truth:
        return 0.0
    top_k_items = retrieved_items[:k]
    gt_set = set(ground_truth)
    hits = sum(1 for item in top_k_items if item in gt_set)
    return float(hits) / float(k)


def reciprocal_rank(retrieved_items: List[Union[int, str]], ground_truth: List[Union[int, str]], k: int = 10) -> float:
    """
    Reciprocal Rank (RR@K): 1 / rank of the first relevant retrieved item within top-K, else 0.0.
    """
    if not ground_truth:
        return 0.0
    gt_set = set(ground_truth)
    for rank, item in enumerate(retrieved_items[:k], start=1):
        if item in gt_set:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved_items: List[Union[int, str]], ground_truth: List[Union[int, str]], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@K) for binary relevance with deduplication.
    Strictly bounded in [0.0, 1.0].
    """
    if not ground_truth:
        return 1.0
    gt_set = set(ground_truth)

    # Compute DCG@K tracking unique relevant hits
    dcg = 0.0
    seen_relevant: Set[Union[int, str]] = set()
    for rank, item in enumerate(retrieved_items[:k], start=1):
        if item in gt_set and item not in seen_relevant:
            seen_relevant.add(item)
            dcg += 1.0 / math.log2(rank + 1)

    # Compute Ideal DCG@K
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return min(1.0, float(dcg / idcg))
