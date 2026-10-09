# Real RBI Regulatory Documents Guide

To give this portfolio project authentic interview credibility, use **real, public RBI regulatory circulars and Master Directions** rather than synthetic or generic text. Real regulatory PDFs have messy headers, multi-tiered numbered clauses, footnotes, and complex cross-references that showcase your parsing and retrieval depth.

---

## Recommended RBI Circulars to Download

Download the official PDFs from the Reserve Bank of India website ([rbi.org.in](https://www.rbi.org.in)) and save them directly in this directory (`data/raw/`):

### 1. Guidelines on Default Loss Guarantee (DLG) in Digital Lending (High Priority)
- **Official Reference:** `RBI/2023-24/41` (DOR.FIN.REC.No.20/03.10.136/2023-24)
- **Date:** June 08, 2023
- **Source Link:** [RBI Notification on DLG in Digital Lending](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12514&Mode=0)
- **Why it's great for RAG testing:**
  - Has strict quantitative limits: **5% cap on total DLG portfolio**.
  - Precise invocation timeline: **Maximum 120 days overdue** before invocation.
  - Clear definitions of Regulated Entities (REs) vs Lending Service Providers (LSPs).
  - Perfect for testing grounded quantitative questions like: *"What is the maximum permissible DLG percentage for an LSP portfolio?"*

### 2. Master Direction – Digital Payment Security Controls
- **Official Reference:** `RBI/2020-21/74` (DoS.CO.CSITE.SEC.No.1852/31.01.015/2020-21)
- **Date:** February 18, 2021
- **Source Link:** [RBI Master Direction on Digital Payment Security Controls](https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12032)
- **Why it's great for RAG testing:**
  - Contains detailed technical architecture mandates: Multi-Factor Authentication (MFA), customer-induced transaction limits, session timeouts, velocity checks.
  - Good for testing technical policy queries like: *"What are the cooling period requirements after password changes?"*

### 3. Master Direction on IT Governance, Risk, Controls and Assurance Practices, 2023
- **Official Reference:** `RBI/2023-24/107` (DoS.CO.CSITE.SEC.No.3/31.01.015/2023-24)
- **Date:** November 07, 2023
- **Source Link:** [RBI Master Direction on IT Governance](https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12562)
- **Why it's great for RAG testing:**
  - Board oversight, CISO reporting line, IT Steering Committee mandates, Cyber Crisis Management Plan (CCMP).

### 4. Reserve Bank of India (Internal Grievance Redress - IGR) Directions, 2024
- **Official Reference:** `RBI/2024-25/110`
- **Date:** 2024
- **Why it's great for RAG testing:**
  - Turnaround times (30 days), Internal Ombudsman escalation matrix.

---

## How to Ingest Once Downloaded

Place the `.pdf` files in this directory (`data/raw/`), then run:

```bash
# Using the project's virtualenv
.venv\Scripts\python scripts/ingest.py --input-dir data/raw
```

The pipeline will parse each document, extract sections with clause numbers and page references, and output structured JSON into `data/processed/`.
