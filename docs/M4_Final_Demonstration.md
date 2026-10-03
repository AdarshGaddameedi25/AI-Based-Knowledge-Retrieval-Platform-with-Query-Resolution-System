# Milestone 4 Final Demonstration Walkthrough

## Step-by-Step Demonstration Guide

1. **Start Backend & Database**:
   ```bash
   alembic upgrade head
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open Frontend**:
   Navigate to `http://localhost:8000` in Google Chrome or Microsoft Edge.

3. **Ingest Three Domain Documents**:
   - Upload `fmla_employee_guide.pdf` (HR domain).
   - Upload `technology_knowledge_base.csv` (Technology domain).
   - Upload `finance_policy.txt` (Finance domain).

4. **Execute Domain-Specific Queries**:
   - **HR**: "What are the eligibility criteria for FMLA leave?" -> Answered with page citations.
   - **Tech**: "How do I configure vector search in PostgreSQL?" -> Answered with chunk evidence.
   - **Finance**: "What is the corporate travel expense reimbursement limit?" -> Answered with policy details.

5. **Test Multi-Turn Context & Pronoun Resolution**:
   - Ask: "What is PostgreSQL?"
   - Follow up: "What are its advantages?" -> Resolved to PostgreSQL's advantages without context bleed.

6. **Test Knowledge Gap Detection**:
   - Ask: "What is the maternity leave policy for Infosys?" -> System returns controlled fallback ("I could not find sufficient information...") and auto-logs a knowledge gap.

7. **Inspect Analytics Dashboard**:
   - Scroll to the **Analytics Dashboard** section or click **Analytics** in the navbar.
   - Observe live KPI cards, domain distributions, top search terms, and the newly tracked knowledge gap.
