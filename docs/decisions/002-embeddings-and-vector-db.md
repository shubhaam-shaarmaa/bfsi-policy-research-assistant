# Architecture Decision Record (ADR) 002: Embeddings & Vector Database for BFSI Policy Research

- **Status:** Accepted
- **Date:** October 2026
- **Context:** BFSI Policy Research Assistant (RAG Pipeline)
- **Author:** Shubham Sharma (AI Engineer / FDE Candidate)

---

## 1. Executive Summary

A production-grade RAG system for a financial institution (banks, NBFCs, asset management firms) cannot treat embeddings and vector databases as black boxes. In banking environments, technical choices are constrained by **regulatory data residency (RBI mandate)**, **statutory auditability**, **cost/latency SLAs**, and **payload filtering precision**.

This document outlines the selection of our vector storage engine (**Qdrant / ChromaDB**) and embedding models (**Local Sentence-Transformers vs. Hosted Voyage/OpenAI**), with quantifiable tradeoffs for technical interviews.

---

## 2. Vector Database Decision: Qdrant vs. ChromaDB

We evaluated the top lightweight and enterprise vector databases for local and self-hosted BFSI deployment:

| Criterion | Qdrant | ChromaDB | Pinecone / Weaviate |
| :--- | :--- | :--- | :--- |
| **Deployment Mode** | In-memory, Embedded disk, or Docker daemon | In-memory or Embedded SQLite+DuckDB | Hosted cloud only (fails air-gapped on-prem banking) |
| **Payload / Metadata Filtering** | **Native Boolean Pre-Filtering** on HNSW graphs | Flat metadata filtering via SQLite | Cloud-dependent |
| **Language & Engine** | Rust (ultra-low latency, zero memory leaks) | Python + C++ (HNSWLib) | Managed SaaS |
| **Idempotent Upsert** | Native deterministic point ID hashing (`UUID5`) | Primary key replacement (`upsert`) | Native ID upsert |
| **Data Residency** | **100% On-Premise / VPC compliant** | **100% On-Premise / VPC compliant** | Risk of cross-border data transfer |

### Our Decision
1. **Primary Enterprise Vector Store: Qdrant**
   - **Why:** Qdrant executes **payload pre-filtering during graph traversal**. In regulatory search, we frequently query with tight filters: `WHERE issuer = 'RBI' AND doc_type = 'circular' AND date >= '2023-01-01'`. 
   - Post-filtering (searching top 100 vectors then discarding non-matching metadata) causes catastrophic recall drop if only 2% of documents match the filter. Qdrant’s custom HNSW implementation checks payload conditions *during* graph exploration.
2. **Local Developer & CI Store: ChromaDB / In-Memory Qdrant**
   - We support an identical interface (`BaseVectorStore`) that allows running unit tests without spinning up Docker, while using the identical storage contract in production.

---

## 3. Embedding Model Decision: Local vs. Hosted

We implemented a pluggable interface (`BaseEmbeddingProvider`) supporting both local open-source models and hosted financial endpoints.

| Model | Provider | Dim | Context | Latency (p95) | Cost / 1M Tokens | BFSI Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`all-MiniLM-L6-v2`** (Default) | Local (HuggingFace) | 384 | 512 tok | **12 ms (CPU)** | **$0.00** | Air-gapped on-premise; high throughput; zero egress. |
| **`BAAI/bge-small-en-v1.5`** | Local (HuggingFace) | 384 | 512 tok | **18 ms (CPU)** | **$0.00** | Top MTEB retrieval performance in the small parameter class. |
| **`voyage-finance-2`** | Hosted (Voyage AI) | 1024 | 32k tok | 180 ms (API) | $0.12 | Fine-tuned on SEC 10-K, earnings calls, financial ontologies. |
| **`text-embedding-3-small`** | Hosted (OpenAI) | 1536 | 8k tok | 150 ms (API) | $0.02 | General purpose; strong semantic clustering. |

### Why We Chose `all-MiniLM-L6-v2` / `bge-small-en` as Default:
1. **Zero Data Egress:** Under the *RBI Master Direction on IT Governance and Information Security*, banks cannot transmit unmasked internal policy drafts or customer SOPs to public SaaS APIs without cryptographic controls and vendor risk assessments. A local model runs entirely within the VPC perimeter.
2. **Dimension Efficiency (384 vs 1536):** A 384-dimensional vector requires **75% less RAM** than 1536 dimensions. For a catalog of 200,000 regulatory clauses, memory footprint drops from ~1.2 GB to ~300 MB, fitting entirely in L3 cache / local RAM.
3. **Pluggable Hosted Fallback:** For public non-sensitive circulars where financial domain nuances dominate, setting `EMBEDDING_PROVIDER=voyage` switches to `voyage-finance-2` with zero code changes.

---

## 4. Re-Ingestion & Idempotency Architecture

A common failure mode in enterprise RAG pipelines is **vector duplication when documents are updated or re-indexed**:
- If an RBI circular is re-ingested after metadata tagging, a naive system inserts another copy of every chunk vector, diluting retrieval scores with duplicate hits.
- **Our Solution:** 
  1. Each chunk content is hashed with SHA-256 (`content_hash`).
  2. The chunk ID is deterministically computed: `{doc_id}#c{chunk_index}`.
  3. Qdrant point IDs are generated via deterministic `UUID5(NAMESPACE_DNS, chunk_id)`.
  4. An `indexed_manifest.json` tracks ingested hashes. If a file is re-processed without text changes, indexing completes in 0 milliseconds. If text changes, `upsert()` overwrites the existing vector in-place.

---

## 5. How to Explain This in an Interview (FDE / AI Engineer)

> **Interviewer:** *"Why did you use a 384-dimensional local model instead of OpenAI's text-embedding-3-large? Doesn't the larger model give better embeddings?"*

> **Your Answer:**
> *"In consumer AI, using OpenAI or Voyage is an easy default. But in BFSI and banking, two real-world constraints dominate: **data residency compliance** and **filtered retrieval latency**.*
> 
> *First, transmitting internal banking SOPs and non-public circulars across public SaaS endpoints creates data privacy compliance risks under RBI guidelines. A local `all-MiniLM-L6-v2` or `bge-small-en` runs completely inside the bank's secure VPC with zero external egress.*
> 
> *Second, 384-dimensional vectors provide a 75% reduction in memory overhead and sub-15ms CPU inference, meaning we don't need expensive GPU clusters just to embed policy updates.*
> 
> *Third, our architecture uses a pluggable `BaseEmbeddingProvider` interface, so if we're dealing with public circulars and want specialized financial ontology tuning, we can toggle to `voyage-finance-2` purely through environment configuration without changing a single line of business logic."*
