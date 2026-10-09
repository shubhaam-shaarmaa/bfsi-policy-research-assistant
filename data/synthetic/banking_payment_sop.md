# [SYNTHETIC TEST DATASET - INTERNAL BANKING SOP - NOT REAL REGULATION]

# Standard Operating Procedure: T+1 Digital Settlement Exception Handling & SSI Break Triage

**Document Reference:** SOP-OPS-2024-V2  
**Effective Date:** January 15, 2024  
**Issuing Authority:** Middle-Office Operations & Settlement Risk Group  

---

## 1. Objective and Overview
This Standard Operating Procedure defines the mandatory escalation hierarchy and resolution protocols for T+1 payment and securities settlement exceptions across commercial banking desks.

## 2. Standard Settlement Instructions (SSI) Discrepancies
2.1 When an automated trade match fails due to an SSI mismatch, the trade status must be set to `BREAK_PENDING_VERIFICATION` within fifteen (15) minutes of ingestion.
2.2 The operational analyst must cross-reference the SWIFT MT541/MT543 message against the golden-source custody database.
2.3 Manual overrides of beneficiary accounts are strictly prohibited without dual authorization from a Senior Operations Manager.

## 3. Priority Break SLAs and DTCC Cut-off Windows
3.1 All high-value trade breaks exceeding USD 1,000,000 equivalent must be triaged within an SLA of two (2) hours from trade capture.
3.2 Trades approaching the 11:30 AM EST DTCC affirmation cut-off window must be flagged with `CRITICAL_AFFIRMATION_DEADLINE`.
3.3 If an affirmative match cannot be established before 11:30 AM EST, the trade must enter bilateral cancellation and re-booking workflows.

## 4. Audit Trail and Record Retention
4.1 All exception tickets, chat transcripts, and audit timestamps shall be retained in immutable cold storage for seven (7) years in compliance with audit directives.
