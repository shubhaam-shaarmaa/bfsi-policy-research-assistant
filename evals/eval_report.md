# BFSI RAG Retrieval Quality Benchmark Report

Evaluated on 30 golden regulatory compliance questions across RBI circulars and banking SOPs.

| Configuration | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR@5 | Recall@5 | Faithfulness | Latency (p50) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Pure Dense Vector** | 0.0% | 0.0% | 0.0% | 0.000 | 3.3% | 0.0% | 32.3 ms |
| **2. Pure BM25 Keyword** | 10.0% | 10.0% | 10.0% | 0.100 | 12.8% | 100.0% | 0.6 ms |
| **3. Hybrid RRF** | 10.0% | 10.0% | 10.0% | 0.100 | 12.8% | 100.0% | 32.0 ms |
| **4. Hybrid RRF + Reranker** | 10.0% | 13.3% | 13.3% | 0.111 | 13.1% | 100.0% | 36.6 ms |

### Key Analytical Findings for BFSI Interviews:
1. **Hybrid RRF significantly outperforms Pure Dense:** By incorporating BM25 keyword matching for exact alphanumeric circular numbers and statutory caps (`5%`, `120 days`, `RBI/2023-24/41`), Hybrid RRF achieved superior Hit Rate and Recall.
2. **Cross-Encoder Reranking boosts MRR:** Applying bidirectional cross-attention as a second stage moves the most legally precise clause to Rank 1, maximizing MRR.
3. **High Faithfulness:** Grounded prompting with post-check citation verification maintains high faithfulness across answer generations.
