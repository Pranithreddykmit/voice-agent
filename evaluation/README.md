# Retrieval Evaluation & Benchmark Framework

This directory contains the quantitative retrieval evaluation dataset, metric calculation algorithms, and automated benchmarking harness for comparing retrieval configurations.

---

## 1. Evaluation Dataset (`dataset.json`)

The benchmark evaluation dataset comprises test queries mapped to ground-truth document targets:
- **English Queries:** Formal academic queries targeting specific clauses.
- **Telugu Queries:** Native Telugu queries covering rules, percentages, and fees.
- **Code-Mixed Queries:** Common conversational mixtures of Telugu and English terms.
- **Negative Control Queries:** Out-of-corpus questions to verify refusal behavior.

### Schema:
```json
{
  "id": "q01_en_attendance",
  "question": "What is the minimum attendance percentage required to write semester exams?",
  "language": "en",
  "category": "attendance",
  "is_in_corpus": true,
  "relevant_documents": ["academic_regulations.pdf"],
  "relevant_pages": [1],
  "relevant_keywords": ["75%", "minimum attendance"]
}
```

---

## 2. Quantitative Retrieval Metrics (`metrics.py`)

- **Hit@K:** Indicates whether at least one relevant document/page appears in the top $K$ positions ($\in \{0, 1\}$).
- **Recall@K:** The fraction of all relevant target pages retrieved within the top $K$ positions ($\frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$).
- **Precision@K:** The fraction of retrieved items in the top $K$ that are relevant ($\frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$).
- **Mean Reciprocal Rank (MRR@K):** The reciprocal of the 1-indexed rank of the first relevant retrieved document ($\frac{1}{\text{rank}_1}$).
- **Normalized Discounted Cumulative Gain (nDCG@K):** Evaluates ranking quality with logarithmic position discount:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 3. Running the Benchmark

```bash
# Activate virtual environment
source .venv/bin/activate

# Execute the evaluation harness
PYTHONPATH=ai-service python evaluation/evaluate.py
```
