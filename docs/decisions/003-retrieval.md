# Architecture Decision Record (ADR) 003: Retrieval Quality — Why Hybrid Search (Dense + BM25) Beats Pure Vector Search in BFSI

- **Status:** Accepted
- **Date:** October 2026
- **Context:** BFSI Policy Research Assistant (RAG Pipeline)
- **Author:** Shubham Sharma (AI Engineer / FDE Candidate)

---

## 1. Executive Summary

A prevailing misconception in modern AI engineering is that dense vector search (semantic similarity) has made classical lexical search (BM25) obsolete. In banking and regulatory compliance (BFSI), **pure dense vector retrieval regularly fails on high-stakes queries**.

Regulatory compliance research demands two orthogonal capabilities:
1. **Conceptual Understanding:** Matching broad semantic intent (e.g. *"What are the vendor governance obligations?"* matches a section titled *"Oversight of Third-Party Lending Service Providers"*).
2. **Pinpoint Lexical Precision:** Exact matches on statutory circular numbers (e.g. `RBI/2023-24/41`), specific clause references (e.g. `Clause 3.1`), statutory acronyms (e.g. `DLG`, `LSP`, `RE`), and compliance figures (`5% cap`, `120 days`).

This document details our evaluation of **Dense Vector Search vs. BM25 vs. Hybrid Reciprocal Rank Fusion (RRF)** on real RBI circulars.

---

## 2. Real-World Failure Modes of Pure Dense Search

### Case 1: Exact Regulatory Circular Numbers & Statutory Citations
- **Query:** `"What did RBI mandate in circular RBI/2023-24/41?"`
- **Dense Vector Failure:** 
  - Tokenizers (`tiktoken`, WordPiece, BPE) shatter `RBI/2023-24/41` into fragments: `['R', 'BI', '/', '202', '3', '-', '24', '/', '41']`.
  - In embedding vector space, `RBI/2023-24/41` is geometrically closer to generic RBI circulars than to the specific Default Loss Guarantee circular. Dense search frequently returns irrelevant circulars from 2023.
- **BM25 Behavior:** 
  - BM25 calculates Inverse Document Frequency (IDF) on exact token sequence `RBI/2023-24/41`. Since this token appears in only one document in the corpus, its IDF score is astronomical.
  - **BM25 Rank: #1 (100% precision). Dense Vector Rank: #8 (misses context).**

---

### Case 2: BFSI Domain Acronyms & Short Abbreviations
- **Query:** `"Can an LSP offer DLG without board approval?"`
- **Dense Vector Failure:**
  - Acronyms like **DLG** (Default Loss Guarantee), **LSP** (Lending Service Provider), and **RE** (Regulated Entity) are 2 to 3 characters long.
  - General embedding models (like `all-MiniLM-L6-v2` or `text-embedding-3-small`) confound `DLG` with generic terms like "dialogue", "digital lending guidelines", or "direct lending".
- **BM25 Behavior:**
  - Direct token match on `DLG` and `LSP` anchors the search directly to the Digital Lending Master Direction.

---

### Case 3: Precise Statutory Thresholds & Numbers
- **Query:** `"What is the maximum portfolio cap for DLG arrangements?"`
- **Dense Vector Failure:**
  - Semantic embeddings understand that "5%", "10%", and "20%" all belong to the semantic cluster of "percentages/limits", but cannot prioritize `5%` over `10%`.
  - If a neighboring clause mentions a "10% capital adequacy ratio", vector distance between the query and the wrong clause is nearly identical to the correct clause.
- **Hybrid RRF Behavior:**
  - BM25 provides the hard lexical constraint (`cap`, `portfolio`, `5%`), while Dense search provides the semantic bridge (`arrangements`, `limits`).

---

## 3. Why Reciprocal Rank Fusion (RRF) Over Score Interpolation

When combining Dense and Sparse (BM25) search, two strategies exist:

### Strategy A: Linear Score Interpolation (Anti-Pattern)
$$Score = \alpha \cdot Score_{dense} + (1 - \alpha) \cdot Score_{bm25}$$
- **Why this fails in production:**
  - Cosine similarity scores are bounded in $[0.0, 1.0]$ with small variances (e.g. $0.78$ vs $0.82$).
  - BM25 scores are unbounded positive numbers dependent on document length and term frequency (e.g. $3.2$ in small corpora, $42.6$ in large corpora).
  - Normalizing BM25 scores via Min-Max makes the retrieval distribution brittle: adding a single outlier document rescales all existing scores.

### Strategy B: Reciprocal Rank Fusion (Our Architecture)
$$RRF\_Score(d) = \sum_{m \in \{dense, bm25\}} \frac{w_m}{k + Rank_m(d)}$$
- **Why this is superior:**
  - **Scale-Invariant:** Uses ordinal rankings ($1, 2, 3\dots$) rather than raw floating-point scores. No calibration or normalization required.
  - **Smoothing Constant ($k = 60$):** Dampens the penalty between ranks (e.g. rank 1 vs rank 2 difference is $1/61 - 1/62 = 0.00026$), preventing a single high-ranking outlier from dominating the pool.
  - **Consensus Reward:** A document ranked #2 in Dense and #2 in BM25 will beat a document ranked #1 in Dense but missing completely from BM25.

---

## 4. The Two-Stage Pipeline: Hybrid Retrieval + Cross-Encoder Reranking

```
User Query
   │
   ├───► Dense Vector Search (Top 20 candidates) ──┐
   │                                                ├──► RRF Fusion (Top 15) ──► Cross-Encoder Reranker ──► Top 5 Chunks
   └───► BM25 Keyword Search (Top 20 candidates) ──┘                                (Deep Cross-Attention)       (To LLM)
```

1. **Stage 1 (High Recall):** Hybrid RRF fuses top 20 dense and top 20 BM25 hits to guarantee that neither semantic nuances nor exact statutory codes are missed.
2. **Stage 2 (High Precision):** A cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`) feeds `(query, passage)` jointly through all transformer layers, capturing cross-token dependencies that bi-encoders cannot see.

---

## 5. How to Explain This in an Interview (FDE / AI Engineer)

> **Interviewer:** *"Why did you implement BM25 and Hybrid search instead of sticking with standard vector search?"*

> **Your Answer:**
> *"In consumer Q&A, pure vector search is often sufficient. But in BFSI compliance, user queries are dominated by exact alphanumeric strings: circular numbers like `RBI/2023-24/41`, statutory acronyms like `DLG` and `LSP`, and specific percentages like `5% cap`.*
> 
> *Vector embeddings frequently fail on these exact tokens because byte-pair tokenizers fragment numbers and short codes across multiple tokens, blurring their distinct identity.*
> 
> *To solve this, I built a **Hybrid Retrieval Engine** combining BM25 keyword search with dense vector search using **Reciprocal Rank Fusion (RRF)**. RRF is scale-invariant, meaning we don't have to calibrate mismatched score distributions between BM25 and cosine similarity.*
> 
> *For high-precision queries, this is passed to an optional **Cross-Encoder Reranker**, ensuring our top 5 retrieved chunks contain both the exact statutory clause and the surrounding legal context."*
