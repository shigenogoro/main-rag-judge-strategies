# Exploring Alternative Judge Strategies for Multi-Agent Filtering in RAG

CS 646 (Information Retrieval) final project, UMass Amherst, Fall 2025.
Full report: [Reports/CS_646_Final_Report_final.pdf](Reports/CS_646_Final_Report_final.pdf)

## Summary

> **Research question:** how do different Judge thresholding strategies affect the trade-off between answer accuracy, efficiency, and cost in MAIN-RAG?

**Background.** [MAIN-RAG](https://arxiv.org/abs/2501.00332) filters noisy retrieved documents with three LLM agents:

| Agent | Role |
|---|---|
| 🔮 **Predictor** | Drafts an answer from each retrieved document on its own |
| ⚖️ **Judge** | Scores each (question, document, answer) triple by how much it prefers "Yes" over "No" |
| ✍️ **Generator** | Writes the final answer from the documents that pass the threshold |

**What we changed.** We kept the LLM Judge score and swapped in different **rule-based thresholding strategies** for deciding which documents reach the Generator:

| Strategy | Threshold τ on per-query judge scores |
|---|---|
| Mean–SD (MAIN-RAG default) | τ = μ − nσ, n ∈ {0.5, 1.0, 1.5} |
| Quantile | τ = q-quantile, q ∈ {0.5, 0.8} (keep top 50% / 20%) |
| Otsu | τ that maximizes between-class variance (parameter-free) |

**What we found.**

- ✅ All of these training-free rules suppress noisy context.
- 📈 On HotpotQA, the rate at which the gold answer appears in the output (GiP) roughly **doubles** compared with Vanilla RAG. With Mistral-7B and Quantile q=0.8, GiP rises from **7.1% to 24.3%**.
- 💪 A stronger Judge (Llama-3.1-8B) handles aggressive filtering well. A weaker Judge (Mistral-7B) starts dropping relevant documents.

## Pipeline

```
query ──► BM25 (Pyserini, k1=0.9, b=0.4) ──► top-K docs
            │
            ▼  for each doc d_i
        Predictor:  LLM answers q using only d_i          → a_i
        Judge:      s_i = log p(" Yes") − log p(" No")    (next-token logits, no sampling)
            │
            ▼
        Threshold:  keep {d_i : s_i ≥ τ}   (Mean–SD / Quantile / Otsu)
            │
            ▼
        Generator:  LLM answers q using the kept docs     → final answer
```

Baselines: **LLM-only** (no context) and **Vanilla RAG** (all K docs, no judge).

## Experimental setup (report)

- **Datasets (BEIR, test split):** HotpotQA, FEVER, FiQA, NQ. Each is subsampled to N = 70 queries, with K = 20 docs per query.
- **Models:** Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct, 4-bit quantized and run with MLX on an M4 Max (64 GB).
- **Decoding:** greedy, at most 64 new tokens, at most 768 input tokens.
- **Metrics:**
  - Retrieval: Precision, Recall, F1, and nDCG, each computed at the mean retained set size K̄ᵣ.
  - QA (HotpotQA): Exact Match, ROUGE-L, and **GiP** (gold-in-prediction containment).

## Key results

**Retrieval filtering, HotpotQA (F1@K̄ᵣ / K̄ᵣ):**

| Method | Mistral-7B | Llama-3.1-8B |
|---|---|---|
| No filter (K=20) | 0.125 / 20 | 0.125 / 20 |
| Mean–SD n=0.5 | 0.132 / 14.4 | 0.160 / 13.9 |
| Otsu | 0.101 / 11.7 | 0.187 / 10.7 |
| Quantile q=0.8 | 0.112 / 6.3 | **0.194 / 5.1** |

**Answer quality, HotpotQA (GiP %).** Exact Match was 0 for every method and is omitted.

| Method | Mistral-7B | Llama-3.1-8B |
|---|---|---|
| LLM-only | 0.0 | 2.9 |
| Vanilla RAG | 7.1 | 8.6 |
| Mean–SD (any n) | 14.3 | 12.9 |
| Otsu | 18.6 | 12.9 |
| Quantile q=0.8 | **24.3** | **15.7** |

Takeaways:
- A stronger Judge (Llama-3.1-8B) tolerates aggressive filtering. A weaker Judge (Mistral) loses relevant documents when filtering is strict, which lowers retrieval F1.
- Mean–SD variants barely differ from one another. The clear gains come from pruning down to about 5–10 documents.
- FiQA gains are small because relevance labels are sparse and many retrieved documents are only marginally relevant.
- Absolute QA scores are low because the 7B/8B generators are small. The improvement shows up mainly in GiP rather than EM.

See the report for FEVER, FiQA, and NQ results, error analysis, and future work.

## Code: [MAIN-RAG.py](MAIN-RAG.py)

| Component | What it does |
|---|---|
| `BeirBM25Runner` / `run_beir_bm25()` | Downloads a BEIR dataset, builds a Pyserini Lucene index, and retrieves with BM25. It then writes a TREC run file and an LLM cache JSON (`work/<dataset>/runs/<dataset>__test__<tag>__llm_cache.json`) containing queries, gold answers, and the retrieved document texts. |
| `Evaluator` | Loads a Hugging Face causal LM and runs the LLM-only, Vanilla RAG, and MAIN-RAG pipelines. |
| `Evaluator._judge_single()` | Computes the judge score: the Yes/No log-prob margin of the next token. |
| `Evaluator.run_main_rag()` | Runs the Predictor, then the Judge, then the Mean–SD, Quantile, and Otsu thresholds, then the Generator. |
| `Evaluator.plot_threshold()` | Saves judge-score histograms with the cutoff line to `Results/Judges/`. |
| `Evaluator.save_results()` | Writes Exact Match per pipeline to `Results/<dataset>_<org>.csv`. |

### Run

Requirements: Python 3.10+, Java 21 for Pyserini, and a GPU (CUDA or Apple MPS) for the LLMs.

```bash
pip install beir pyserini pytrec_eval torch transformers huggingface_hub pandas matplotlib tqdm
export HF_TOKEN=hf_xxx          # needed for gated models (Llama, Gemma)
python MAIN-RAG.py
```

Edit the `__main__` block to choose `dataset_names`, `model_names`, `max_queries`, `n_std`, and `top_k_docs`. The quantile `q` defaults to 0.8. To change it, set `evaluator.quantile_q`.

> Note: this script is the PyTorch/transformers version of the pipeline. The report's numbers come from 4-bit MLX runs of Mistral-7B and Llama-3.1-8B. The script's current defaults (`gemma-2-27b-it`, 50 queries, n=1.5) differ from that setup, and it computes Exact Match only. ROUGE-L, GiP, and the retrieval metrics in the tables came from separate evaluation.

## Repository layout

```
MAIN-RAG.py                    # BM25 retrieval + MAIN-RAG judge/threshold pipeline
CS_646_Project_Baseline.ipynb  # early baselines (Vanilla RAG, BM25 score cutoff, DPR)
Reports/                       # proposal and final report (PDF)
Milestones/                    # milestone submissions
```

## Team

Sheng-Kai Wen, Frances Chang, Mohammadreza Teymoorianfard, Albi Marini, Mohammad Derakhshi

## Reference

Chang et al. 2024. *MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation.* arXiv:2501.00332.
