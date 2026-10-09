# Deployment Guide — BFSI Policy Research Assistant

This guide covers options for deploying the BFSI Policy Research Assistant across local containers, sovereign VPCs, and cloud environments.

---

## 1. Local / On-Prem Deployment via Docker Compose

The simplest way to run both backend and frontend in production mode:

```bash
# Clone the repository
git clone https://github.com/shubhaam-shaarmaa/bfsi-policy-research-assistant.git
cd bfsi-policy-research-assistant

# Optional: Add your Anthropic API Key for live Claude 3.5 Sonnet generation
# (If omitted, system gracefully falls back to deterministic Mock LLM)
export ANTHROPIC_API_KEY="your-api-key"

# Build and start services
docker-compose up --build -d
```

### Accessing the Services
- **React Frontend:** [http://localhost:5173](http://localhost:5173)
- **FastAPI REST API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 2. Cloud Deployment on Render (1-Click Blueprint)

The repository includes a ready-to-use `render.yaml` specification:

1. Fork or push this repository to GitHub.
2. Sign in to [Render](https://render.com).
3. Click **Blueprints** → **New Blueprint Instance**.
4. Connect the `bfsi-policy-research-assistant` repository.
5. Render will automatically configure:
   - A Python web service running FastAPI (`/health` monitoring enabled).
   - A static site serving the React 19 frontend bundle.
6. Click **Apply** to deploy.

---

## 3. Sovereign Cloud / VPC Deployment (Banking Production)

In accordance with **RBI Master Directions on Information Technology Governance (2023)** and the **DPDP Act 2023**:

1. **VPC Placement:** Deploy both backend and vector storage within private subnets in an Indian AWS region (`ap-south-1` Mumbai / `ap-south-2` Hyderabad) or Indian Azure/GCP data centers.
2. **Persistent Volumes:** Mount high-IOPS EBS volumes (`gp3`) to `/app/data/vectorstore` and `/app/data/processed` to ensure vector indexes persist across pod restarts.
3. **Internal Load Balancer:** Expose the FastAPI service via an internal Network Load Balancer (NLB) with TLS termination.
4. **Zero Data Egress:** Set `EMBEDDING_PROVIDER=local` (`sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-large-en-v1.5`) and run LLM inference on private VPC Triton/vLLM endpoints to guarantee zero outbound cross-border data egress.

---

## 4. Continuous Integration (CI)

A GitHub Actions workflow is configured in `.github/workflows/ci.yml`:
- Runs all 33 unit and integration tests across ingestion, chunking, retrieval, generation, and API endpoints.
- Verifies that the React Vite frontend builds cleanly with zero errors.
- Triggers on every push to `main` and pull requests.
