"""
Unit tests for Retrieval Evaluation Metrics.
"""

import pytest
import math
from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    ndcg_at_k
)


def test_hit_at_k():
    retrieved = [1, 2, 3, 4, 5]
    assert hit_at_k(retrieved, [1], k=1) == 1.0
    assert hit_at_k(retrieved, [3], k=1) == 0.0
    assert hit_at_k(retrieved, [3], k=3) == 1.0
    assert hit_at_k(retrieved, [99], k=5) == 0.0


def test_recall_at_k():
    retrieved = [1, 2, 3, 4, 5]
    assert recall_at_k(retrieved, [1, 2], k=2) == 1.0
    assert recall_at_k(retrieved, [1, 2, 99], k=3) == 2.0 / 3.0
    assert recall_at_k(retrieved, [99], k=5) == 0.0


def test_precision_at_k():
    retrieved = [1, 2, 3, 4, 5]
    assert precision_at_k(retrieved, [1], k=1) == 1.0
    assert precision_at_k(retrieved, [1], k=2) == 0.5
    assert precision_at_k(retrieved, [1, 2], k=4) == 2.0 / 4.0


def test_reciprocal_rank():
    retrieved = [10, 20, 30, 40]
    assert reciprocal_rank(retrieved, [10]) == 1.0
    assert reciprocal_rank(retrieved, [20]) == 0.5
    assert reciprocal_rank(retrieved, [30]) == 1.0 / 3.0
    assert reciprocal_rank(retrieved, [99]) == 0.0


def test_ndcg_at_k():
    # Rank 1 hit: NDCG@3 should be 1.0
    assert ndcg_at_k([1, 2, 3], [1], k=3) == 1.0

    # Rank 2 hit: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0 => NDCG = 1/log2(3)
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(ndcg_at_k([2, 1, 3], [1], k=3) - expected_ndcg) < 1e-5

    # No hits: NDCG should be 0.0
    assert ndcg_at_k([2, 3, 4], [1], k=3) == 0.0
