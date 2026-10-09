"""FastAPI Application for BFSI Policy Research Assistant.

Exposes REST endpoints for document ingestion, querying, citation inspection,
health checks, and LLMOps query logging.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import shutil
from typing import Any, Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.api.database import (
    delete_document_record,
    get_query_logs,
    init_db,
    list_documents,
    log_query,
    upsert_document_record,
)
from src.chunking.pipeline import ChunkingPipeline
from src.core.logging import app_logger
from src.core.models import DocumentType
from src.embeddings.factory import get_embedding_provider
from src.generation.generator import GroundedGenerator
from src.generation.models import GroundedAnswer
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.engine import RetrievalEngine
from src.vectorstore.factory import get_vector_store
from src.vectorstore.pipeline import VectorIndexingPipeline


# Global singletons
_embedding_provider = None
_vector_store = None
_retrieval_engine = None
_grounded_generator = None

# Ensure SQLite schema is ready
init_db()


def get_services():
    global _embedding_provider, _vector_store, _retrieval_engine, _grounded_generator
    if _embedding_provider is None:
        _embedding_provider = get_embedding_provider()
    if _vector_store is None:
        _vector_store = get_vector_store(dimension=_embedding_provider.dimension)
    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine(
            vector_store=_vector_store,
            embedding_provider=_embedding_provider,
        )
    if _grounded_generator is None:
        _grounded_generator = GroundedGenerator()
    return _retrieval_engine, _grounded_generator, _vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    init_db()
    # Initialize services
    get_services()
    app_logger.info("FastAPI service initialized successfully.")
    yield


app = FastAPI(
    title="BFSI Policy Research Assistant API",
    description="Grounded Regulatory Q&A with Citations and Hybrid Retrieval for RBI Circulars",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Regulatory question or compliance query")
    mode: Literal["dense", "bm25", "hybrid"] = Field("hybrid", description="Retrieval mode")
    top_k: int = Field(4, ge=1, le=10, description="Number of source passages to retrieve")
    use_reranker: bool = Field(False, description="Apply Cross-Encoder reranking")
    filters: dict[str, Any] | None = Field(None, description="Optional metadata filter dictionary")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    embedding_provider: str
    vector_db_provider: str
    chunking_strategy: str
    total_documents: int
    total_vectors: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health status and current system topology."""
    _, _, vstore = get_services()
    docs = list_documents()
    total_vecs = vstore.count() if vstore else 0

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        vector_db_provider=settings.VECTOR_DB_PROVIDER,
        chunking_strategy=settings.CHUNKING_STRATEGY,
        total_documents=len(docs),
        total_vectors=total_vecs,
    )


@app.get("/documents")
def get_all_documents():
    """List all registered regulatory documents and their ingestion metadata."""
    return list_documents()


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Upload, parse, chunk, and index a new regulatory document."""
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt, .md",
        )

    # Save to data/raw/
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = settings.DATA_RAW_DIR / file.filename

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Parse document
        ingest_pipeline = IngestionPipeline()
        parsed_doc = ingest_pipeline.ingest_file(destination)

        # 2. Chunk document
        chunk_pipeline = ChunkingPipeline()
        chunks = chunk_pipeline.chunk_document(parsed_doc)

        # 3. Vectorize and index
        engine, _, vstore = get_services()
        index_pipeline = VectorIndexingPipeline(vector_store=vstore, embedding_provider=_embedding_provider)
        index_pipeline.index_chunks(chunks, force=True)

        # 4. Refresh BM25
        engine._sync_bm25_index()

        # 5. Record in SQLite
        upsert_document_record(
            doc_id=parsed_doc.metadata.doc_id,
            title=parsed_doc.metadata.title,
            issuer=parsed_doc.metadata.issuer,
            circular_number=parsed_doc.metadata.circular_number,
            date=parsed_doc.metadata.date,
            doc_type=parsed_doc.metadata.doc_type.value
            if isinstance(parsed_doc.metadata.doc_type, DocumentType)
            else str(parsed_doc.metadata.doc_type),
            source_filename=parsed_doc.metadata.source_filename,
            total_pages=parsed_doc.page_count,
            total_chunks=len(chunks),
        )

        return {
            "status": "success",
            "message": f"Successfully ingested and indexed '{file.filename}'",
            "doc_id": parsed_doc.metadata.doc_id,
            "title": parsed_doc.metadata.title,
            "circular_number": parsed_doc.metadata.circular_number,
            "total_pages": parsed_doc.page_count,
            "total_chunks": len(chunks),
        }

    except Exception as e:
        app_logger.error("Error processing document %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document, its vector embeddings, and chunk metadata."""
    engine, _, vstore = get_services()
    if vstore:
        vstore.delete_document(doc_id)

    # Delete chunks json
    chunk_file = settings.DATA_CHUNKS_DIR / f"chunks_{doc_id}.json"
    if chunk_file.exists():
        chunk_file.unlink()

    # Delete processed doc json
    proc_file = settings.DATA_PROCESSED_DIR / f"{doc_id}.json"
    if proc_file.exists():
        proc_file.unlink()

    # Remove from SQLite
    deleted = delete_document_record(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    # Refresh BM25
    engine._sync_bm25_index()

    return {"status": "success", "message": f"Document '{doc_id}' deleted successfully."}


@app.post("/ask", response_model=GroundedAnswer)
def ask_question_endpoint(req: AskRequest):
    """Answer a regulatory question using grounded RAG with verified citations."""
    engine, generator, _ = get_services()

    # 1. Retrieve candidates
    search_hits = engine.search(
        query=req.query,
        mode=req.mode,
        use_reranker=req.use_reranker,
        top_k=req.top_k,
        filters=req.filters,
    )

    if not search_hits:
        return GroundedAnswer(
            query=req.query,
            answer="The provided regulatory documents do not contain information to answer this question.",
            citations=[],
            confidence_note="No relevant passages matched the query.",
            has_sufficient_context=False,
            retrieved_chunks=[],
            model_used=settings.LLM_MODEL,
        )

    # 2. Synthesize Grounded Answer
    grounded_ans = generator.generate_answer(query=req.query, retrieved_results=search_hits)

    # 3. Log query to SQLite audit trail
    chunk_ids = [c.get("chunk_id", "") for c in grounded_ans.retrieved_chunks]
    log_query(
        query_text=req.query,
        answer_text=grounded_ans.answer,
        model_used=grounded_ans.model_used,
        latency_ms=grounded_ans.latency_ms,
        prompt_tokens=grounded_ans.prompt_tokens,
        completion_tokens=grounded_ans.completion_tokens,
        retrieved_chunk_ids=chunk_ids,
        citations_count=len(grounded_ans.citations),
        verification_passed=grounded_ans.verification_passed,
        has_sufficient_context=grounded_ans.has_sufficient_context,
        confidence_note=grounded_ans.confidence_note,
    )

    return grounded_ans


@app.get("/queries")
def get_audit_query_logs(limit: int = 50):
    """Retrieve historical query logs for LLMOps, latency tracking, and evaluation."""
    return get_query_logs(limit=limit)
