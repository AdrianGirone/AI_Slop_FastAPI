# AI Job Agent - Implementation Backlog

**Last Updated:** November 6, 2025
**Project Status:** 30% Complete (Foundation Ready)
**Tech Stack:** FastAPI + LangGraph + Ollama (llama3.1:70b) + ChromaDB + Playwright + Jinja2

---

## 📊 Current State Assessment

### ✅ What's Complete (Foundation - 30%)

1. **FastAPI Application Structure**
   - Async/await throughout
   - Pydantic models and settings
   - CORS middleware
   - Health check endpoint
   - Logging configuration

2. **Ollama Integration**
   - Async httpx client (`utils/llm.py`)
   - Query and streaming support
   - Connection health checks
   - Error handling

3. **Basic API Endpoints**
   - `GET /` - Web UI (Jinja2 template)
   - `POST /api/submit` - Text prompt endpoint
   - `GET /api/ask` - Query endpoint
   - `GET /health` - Ollama connectivity check

4. **Development Environment**
   - Professional venv/conda setup
   - All dependencies declared in requirements.txt
   - Environment-based configuration (.env)
   - Clean project structure (15 files)

### ❌ What's Missing (Core Features - 70%)

1. **RAG Pipeline** - Resume/JD semantic search
2. **Job Search Tools** - Multi-source aggregation
3. **AI Agents** - LangGraph state machines
4. **Resume Tailoring** - Core value proposition
5. **Application Tracking** - Database persistence
6. **Background Tasks** - Long-running operations
7. **Testing Suite** - Quality assurance
8. **Security Layer** - Encryption, rate limiting

---

## 🎯 Strategic Direction

### Core Value Proposition

**Enable users to:**

1. Upload their resume (PDF/DOCX)
2. Search jobs from multiple sources
3. Get AI-tailored resumes for specific job descriptions
4. Track applications and their status
5. (Future) Semi-automate applications with human approval

### Technology Choices (Confirmed)

- **Backend Framework:** FastAPI (async-first)
- **LLM Orchestration:** LangGraph (state machines, not CrewAI)
- **Local LLM:** Ollama (llama3.1:70b, 128k context)
- **Vector Store:** ChromaDB (persistent, with encryption)
- **Web Automation:** Playwright (ethical scraping only)
- **Templates:** Jinja2 (server-side rendering)
- **Database:** SQLModel (to add - for job tracking)
- **Background Tasks:** FastAPI BackgroundTasks → ARQ (if needed)

### Architecture Philosophy

1. **Start Simple, Add Complexity as Needed**
   - Build single-agent LangGraph chains first
   - Add multi-agent orchestration only if required
   - Avoid premature optimization

2. **Human-in-the-Loop by Default**
   - User reviews tailored resumes before applying
   - Explicit approval required for any automation
   - Show diffs, not just final output

3. **Privacy-First**
   - All data local (no cloud LLMs)
   - Encrypt resume storage
   - No telemetry without consent

4. **Ethical Automation**
   - Respect robots.txt
   - Rate limiting on all scrapers
   - Prefer APIs over scraping
   - User approval before auto-apply

---

## 🚀 Implementation Roadmap

### Phase 1: RAG Foundation (Priority: CRITICAL | Est: 2-3 days)

**Goal:** Enable resume understanding and semantic search

#### Task 1.1: Document Processing Pipeline

**File:** `utils/rag.py`

```python
Features to implement:
- PDF/DOCX loaders (pypdf2, python-docx)
- Text chunking (RecursiveCharacterTextSplitter, 1000 chunks, 200 overlap)
- Ollama embeddings integration
- ChromaDB persistent storage with encryption
- Metadata extraction (skills, experience, education)
```

**Why this matters:** Resume tailoring is impossible without understanding resume content semantically.

**Acceptance Criteria:**

- [ ] Can upload PDF resume and store in ChromaDB
- [ ] Can query: "What are my Python skills?" → Retrieves relevant chunks
- [ ] Can query: "What's my experience level?" → Understands context
- [ ] Embeddings persist across restarts
- [ ] Collections are encrypted at rest

**Dependencies:**

```python
# Already in requirements.txt:
chromadb==0.4.22
pypdf2==3.0.1
python-docx==1.1.0
docx2txt==0.8
```

**Additional needs:**

```python
cryptography==41.0.7  # For resume encryption
```

#### Task 1.2: Resume Upload Endpoint

**File:** `app/routers/resume.py` (new)

```python
Endpoints to create:
- POST /api/v1/resume/upload - Upload resume file
- GET /api/v1/resume/query - Query resume content
- GET /api/v1/resume/skills - Extract structured skills
- DELETE /api/v1/resume/{id} - Delete resume
```

**Why this matters:** Users need a way to get their resume into the system.

**Acceptance Criteria:**

- [ ] Supports PDF and DOCX upload
- [ ] Returns resume_id for future reference
- [ ] Validates file size (max 10MB) and type
- [ ] Extracts metadata (name, email, phone)
- [ ] Stores in ChromaDB with unique collection per user

#### Task 1.3: Job Description Analysis

**File:** `utils/jd_parser.py` (new)

```python
Features to implement:
- JD text extraction and cleaning
- Requirements extraction (must-have vs nice-to-have)
- Skills identification
- Experience level detection
- Location/remote parsing
```

**Why this matters:** Tailoring requires understanding what the job needs.

**Acceptance Criteria:**

- [ ] Can parse JD text into structured format
- [ ] Identifies technical skills (Python, FastAPI, etc.)
- [ ] Extracts years of experience required
- [ ] Determines if role is remote/hybrid/onsite
- [ ] Handles various JD formats

---

### Phase 2: Job Search Integration (Priority: HIGH | Est: 2-3 days)

**Goal:** Aggregate jobs from multiple sources

#### Task 2.1: Job Search Tools

**File:** `tools/job_search.py` (new)

**Multi-Source Strategy:**

**Tier 1 - Public APIs (Preferred):**

```python
1. RemoteOK API (remote tech jobs, free, no key)
   - https://remoteok.com/api
   - Best for: Remote tech positions

2. Adzuna API (all jobs, 250/month free)
   - https://api.adzuna.com/v1/api/jobs/us/search
   - Best for: General job search with location

3. GitHub Jobs API (deprecated but replacements exist)
   - Consider: https://www.themuse.com/developers/api/v2
```

**Tier 2 - Ethical Scraping:**

```python
4. LinkedIn Public Job Listings
   - Use Playwright, respect robots.txt
   - Rate limit: 1 request per 5 seconds

5. Standard ATS Systems (Greenhouse, Lever)
   - These have predictable structure
   - Companies: /careers page
```

**Tier 3 - Fallback:**

```python
6. Company Career Pages
   - Requires custom selectors per site
   - Maintenance heavy, use sparingly
```

**Implementation Approach:**

```python
from langchain.tools import tool
from playwright.async_api import async_playwright

@tool
async def search_remoteok(query: str, limit: int = 20) -> list[dict]:
    """Search RemoteOK for remote tech jobs."""
    # Implement API call
    pass

@tool
async def search_adzuna(query: str, location: str = "remote") -> list[dict]:
    """Search Adzuna with API key."""
    # Implement with rate limiting
    pass

@tool
async def scrape_greenhouse(company_url: str) -> list[dict]:
    """Scrape Greenhouse ATS (ethical, standard format)."""
    async with async_playwright() as p:
        # Respect robots.txt
        # Add random delays (2-5 seconds)
        # Extract jobs
        pass
```

**Acceptance Criteria:**

- [ ] Can search RemoteOK API
- [ ] Can search Adzuna API (with quota tracking)
- [ ] Can scrape at least 2 standard ATS systems
- [ ] All scrapers respect rate limits
- [ ] Returns standardized job schema
- [ ] Handles network errors gracefully

#### Task 2.2: Job Search Endpoint

**File:** `app/routers/jobs.py` (new)

```python
Endpoints to create:
- POST /api/v1/jobs/search - Search jobs across sources
- GET /api/v1/jobs/{id} - Get job details
- POST /api/v1/jobs/{id}/save - Save job for later
- GET /api/v1/jobs/saved - Get saved jobs
```

**Acceptance Criteria:**

- [ ] Aggregates results from multiple sources
- [ ] Deduplicates jobs (same company + title)
- [ ] Caches results (24h TTL)
- [ ] Returns jobs in standard format
- [ ] Supports pagination

#### Task 2.3: Job Matching Score

**File:** `utils/matching.py` (new)

```python
Features:
- Calculate resume-to-JD similarity score
- Use embeddings for semantic matching
- Factor in: skills match, experience level, location
- Rank jobs by relevance
```

**Why this matters:** Users shouldn't manually review 100 jobs. Show best matches first.

**Acceptance Criteria:**

- [ ] Calculates 0-100 match score
- [ ] Factors in required vs. nice-to-have skills
- [ ] Considers experience level fit
- [ ] Returns explanation of score
- [ ] Can sort jobs by match score

---

### Phase 3: LangGraph Agent System (Priority: CRITICAL | Est: 3-4 days)

**Goal:** Build intelligent resume tailoring with LangGraph state machines

#### Task 3.1: LangGraph Setup

**File:** `agents/graph.py` (new)

**Architecture Decision: LangGraph over CrewAI**

**Why LangGraph:**

- More control over state management
- Better for async operations
- Cleaner debugging (visual graph)
- No heavyweight abstractions

**State Machine Design:**

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class TailoringState(TypedDict):
    resume_id: str
    job_description: str
    job_requirements: list[str]  # Extracted from JD
    resume_context: str          # Retrieved from RAG
    matched_skills: list[str]    # Skills in both resume + JD
    missing_skills: list[str]    # Skills in JD but not resume
    tailored_bullets: list[str]  # AI-generated bullet points
    tailored_summary: str        # AI-generated summary
    quality_score: float         # Hallucination check score
    final_resume: str            # DOCX-ready markdown

def create_tailoring_graph() -> StateGraph:
    workflow = StateGraph(TailoringState)

    # Nodes (steps in the process)
    workflow.add_node("parse_jd", parse_job_description)
    workflow.add_node("retrieve_context", retrieve_resume_context)
    workflow.add_node("match_skills", match_skills_to_jd)
    workflow.add_node("tailor_bullets", generate_tailored_bullets)
    workflow.add_node("tailor_summary", generate_tailored_summary)
    workflow.add_node("quality_check", check_for_hallucinations)
    workflow.add_node("format_output", format_as_docx)

    # Edges (flow control)
    workflow.set_entry_point("parse_jd")
    workflow.add_edge("parse_jd", "retrieve_context")
    workflow.add_edge("retrieve_context", "match_skills")
    workflow.add_edge("match_skills", "tailor_bullets")
    workflow.add_edge("tailor_bullets", "tailor_summary")
    workflow.add_edge("tailor_summary", "quality_check")
    workflow.add_conditional_edges(
        "quality_check",
        lambda state: "format_output" if state["quality_score"] > 0.8 else "tailor_bullets",
        {
            "format_output": "format_output",
            "tailor_bullets": "tailor_bullets"  # Retry if quality low
        }
    )

    return workflow.compile()
```

**Acceptance Criteria:**

- [ ] LangGraph state machine compiles
- [ ] Can visualize graph with mermaid
- [ ] State persists between steps
- [ ] Can retry failed steps
- [ ] Supports async execution

#### Task 3.2: Tailoring Agent Implementation

**File:** `agents/tailor_agent.py` (new)

**Core Logic:**

```python
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

async def parse_job_description(state: TailoringState) -> TailoringState:
    """Extract structured requirements from JD."""
    llm = ChatOllama(model="llama3.1:70b", temperature=0.1)

    prompt = ChatPromptTemplate.from_template("""
    Analyze this job description and extract:
    1. Required technical skills
    2. Required experience level (years)
    3. Nice-to-have skills
    4. Key responsibilities

    Job Description:
    {jd}

    Return as JSON.
    """)

    result = await llm.ainvoke(prompt.format(jd=state["job_description"]))
    requirements = parse_json(result.content)

    state["job_requirements"] = requirements
    return state

async def retrieve_resume_context(state: TailoringState) -> TailoringState:
    """Use RAG to get relevant resume sections."""
    from utils.rag import get_rag_pipeline

    rag = get_rag_pipeline(state["resume_id"])

    # Query for each requirement
    context_parts = []
    for req in state["job_requirements"]:
        relevant = rag.query(f"Experience or projects related to: {req}")
        context_parts.append(relevant)

    state["resume_context"] = "\n\n".join(context_parts)
    return state

async def match_skills_to_jd(state: TailoringState) -> TailoringState:
    """Identify skill overlaps and gaps."""
    llm = ChatOllama(model="llama3.1:70b", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Given resume context and job requirements, identify:
    1. Skills the candidate HAS that match the job
    2. Skills the job requires that candidate is MISSING

    Resume Context:
    {context}

    Job Requirements:
    {requirements}

    CRITICAL: Only include skills explicitly mentioned in resume.
    Do NOT invent or assume skills.

    Return as JSON: {{"matched": [...], "missing": [...]}}
    """)

    result = await llm.ainvoke(prompt.format(
        context=state["resume_context"],
        requirements=state["job_requirements"]
    ))

    skills = parse_json(result.content)
    state["matched_skills"] = skills["matched"]
    state["missing_skills"] = skills["missing"]
    return state

async def generate_tailored_bullets(state: TailoringState) -> TailoringState:
    """Rewrite resume bullets to emphasize job-relevant skills."""
    llm = ChatOllama(model="llama3.1:70b", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    Rewrite resume bullet points to emphasize skills needed for this job.

    Resume Context:
    {context}

    Matched Skills to Highlight:
    {matched_skills}

    Job Requirements:
    {requirements}

    RULES:
    1. ONLY use experiences from the resume context
    2. DO NOT invent projects, skills, or accomplishments
    3. Emphasize quantitative results (X% improvement, $Y saved)
    4. Use action verbs (Developed, Implemented, Optimized)
    5. Keep bullets under 150 characters

    Generate 5-7 tailored bullet points.
    """)

    result = await llm.ainvoke(prompt.format(
        context=state["resume_context"],
        matched_skills=state["matched_skills"],
        requirements=state["job_requirements"]
    ))

    state["tailored_bullets"] = result.content.split("\n")
    return state

async def check_for_hallucinations(state: TailoringState) -> TailoringState:
    """Verify tailored content doesn't invent facts."""
    llm = ChatOllama(model="llama3.1:70b", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Compare the tailored resume bullets to the original resume context.

    Original Resume:
    {context}

    Tailored Bullets:
    {bullets}

    For each tailored bullet, verify:
    1. Is this experience mentioned in the original resume? (YES/NO)
    2. Are the quantitative claims accurate? (YES/NO)
    3. Are the technical skills mentioned actually in the original? (YES/NO)

    Return quality score (0-1) and list of any hallucinated claims.
    """)

    result = await llm.ainvoke(prompt.format(
        context=state["resume_context"],
        bullets=state["tailored_bullets"]
    ))

    quality = parse_quality_check(result.content)
    state["quality_score"] = quality["score"]

    return state
```

**Acceptance Criteria:**

- [ ] Can parse JD into structured requirements
- [ ] Retrieves relevant resume context via RAG
- [ ] Identifies skill matches accurately
- [ ] Generates tailored bullets without hallucinations
- [ ] Quality check catches invented experiences
- [ ] Retries if quality score < 0.8
- [ ] Complete flow takes < 60 seconds

#### Task 3.3: Tailoring API Endpoint

**File:** `app/routers/tailor.py` (new)

```python
from fastapi import BackgroundTasks

Endpoints to create:
- POST /api/v1/tailor/start - Start tailoring job (returns task_id)
- GET /api/v1/tailor/{task_id}/status - Check progress
- GET /api/v1/tailor/{task_id}/result - Get tailored resume
- GET /api/v1/tailor/{task_id}/diff - Show before/after comparison
```

**Why background tasks:** LangGraph execution takes 30-60 seconds. Can't block HTTP request.

**Acceptance Criteria:**

- [ ] Returns immediately with task_id
- [ ] Stores task state in memory (or Redis)
- [ ] Can query status (pending/running/completed/failed)
- [ ] Returns full tailored resume with metadata
- [ ] Shows diff between original and tailored
- [ ] Handles failures gracefully

---

### Phase 4: Database & Persistence (Priority: HIGH | Est: 1-2 days)

**Goal:** Track applications, resume versions, job history

#### Task 4.1: Database Models

**File:** `app/models/db.py` (new)

```python
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional

class Resume(SQLModel, table=True):
    id: int = Field(primary_key=True)
    user_id: str  # For multi-user support later
    filename: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
    chroma_collection_id: str  # Reference to ChromaDB
    metadata: dict  # Name, email, skills extracted
    is_active: bool = True

    versions: list["ResumeVersion"] = Relationship(back_populates="resume")

class ResumeVersion(SQLModel, table=True):
    id: int = Field(primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    job_id: Optional[int] = Field(foreign_key="job.id")
    created_at: datetime = Field(default_factory=datetime.now)
    tailored_content: str  # DOCX markdown
    changes_summary: str  # What was changed

    resume: Resume = Relationship(back_populates="versions")
    job: Optional["Job"] = Relationship(back_populates="tailored_resumes")

class Job(SQLModel, table=True):
    id: int = Field(primary_key=True)
    url: str = Field(index=True)
    title: str
    company: str
    description: str
    requirements: dict  # Structured requirements
    location: str
    remote: bool
    salary_range: Optional[str]
    source: str  # "remoteok", "adzuna", etc.
    match_score: Optional[float]
    discovered_at: datetime = Field(default_factory=datetime.now)
    is_saved: bool = False

    applications: list["Application"] = Relationship(back_populates="job")
    tailored_resumes: list["ResumeVersion"] = Relationship(back_populates="job")

class Application(SQLModel, table=True):
    id: int = Field(primary_key=True)
    job_id: int = Field(foreign_key="job.id")
    resume_version_id: int = Field(foreign_key="resumeversion.id")
    applied_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="applied")  # applied, rejected, interviewing, offer
    notes: Optional[str]
    follow_up_date: Optional[datetime]

    job: Job = Relationship(back_populates="applications")
    resume_version: ResumeVersion = Relationship()
```

**Add to requirements.txt:**

```python
sqlmodel==0.0.14
alembic==1.13.1  # Database migrations
```

**Acceptance Criteria:**

- [ ] Database schema defined
- [ ] SQLModel models created
- [ ] Alembic migrations setup
- [ ] Can create/read/update/delete all entities
- [ ] Relationships work correctly
- [ ] Indexes on frequently queried fields

#### Task 4.2: Application Tracking Endpoints

**File:** `app/routers/applications.py` (new)

```python
Endpoints to create:
- POST /api/v1/applications - Record application
- GET /api/v1/applications - List all applications
- GET /api/v1/applications/{id} - Get application details
- PATCH /api/v1/applications/{id} - Update status
- GET /api/v1/applications/stats - Get statistics
```

**Acceptance Criteria:**

- [ ] Can record job application with resume version
- [ ] Can update application status
- [ ] Can query applications by status, date, company
- [ ] Returns stats (total applied, response rate, etc.)
- [ ] Prevents duplicate applications to same job

---

### Phase 5: Production Readiness (Priority: MEDIUM | Est: 2-3 days)

#### Task 5.1: Security Implementation

**Files:** Multiple

**Features to add:**

1. **Resume Encryption** (`utils/encryption.py`)

```python
from cryptography.fernet import Fernet

class EncryptedVectorStore:
    """Wrapper around ChromaDB with encryption."""
    # Encrypt resume text before storing
    # Decrypt on retrieval
```

2. **Rate Limiting** (`app/middleware/rate_limit.py`)

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

# Apply to expensive endpoints:
# /api/v1/tailor/start - 5/minute
# /api/v1/jobs/search - 10/minute
```

3. **Input Validation**

```python
# Strengthen Pydantic models
class ResumeUpload(BaseModel):
    file: UploadFile

    @validator("file")
    def validate_file(cls, v):
        if v.size > 10_000_000:  # 10MB
            raise ValueError("File too large")
        if v.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            raise ValueError("Invalid file type")
        return v
```

4. **Error Handling Middleware**

```python
@app.middleware("http")
async def error_handler(request, call_next):
    try:
        return await call_next(request)
    except OllamaConnectionError:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM service unavailable", "code": "OLLAMA_DOWN"}
        )
    except ChromaDBError:
        return JSONResponse(
            status_code=503,
            content={"error": "Vector store unavailable", "code": "CHROMADB_DOWN"}
        )
    except Exception as e:
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "code": "INTERNAL_ERROR"}
        )
```

**Add to requirements.txt:**

```python
cryptography==41.0.7
slowapi==0.1.9
```

**Acceptance Criteria:**

- [ ] Resume data encrypted at rest
- [ ] Rate limiting on all expensive endpoints
- [ ] File uploads validated (size, type)
- [ ] All errors return structured JSON
- [ ] Sensitive errors don't leak implementation details

#### Task 5.2: Observability

**Files:** `app/middleware/logging.py`, `app/monitoring.py`

**Features:**

1. **Structured Logging**

```python
import structlog

logger = structlog.get_logger()

# Usage:
logger.info("resume_uploaded",
    resume_id=resume_id,
    filename=filename,
    size_bytes=file_size
)

logger.info("tailoring_completed",
    task_id=task_id,
    duration_ms=duration,
    quality_score=score
)
```

2. **Metrics Collection**

```python
# Track:
- Total resumes uploaded
- Total jobs searched
- Total tailoring requests
- Average tailoring time
- Ollama API latency
- ChromaDB query time
- Application conversion rate
```

3. **Health Checks**

```python
@app.get("/health/detailed")
async def health_detailed():
    return {
        "ollama": await check_ollama(),
        "chromadb": await check_chromadb(),
        "database": await check_database(),
        "disk_space": get_disk_space(),
    }
```

**Add to requirements.txt:**

```python
structlog==23.2.0
prometheus-client==0.19.0  # If using Prometheus
```

**Acceptance Criteria:**

- [ ] All operations logged with context
- [ ] Can query logs by task_id, user_id, etc.
- [ ] Detailed health endpoint shows all services
- [ ] Metrics exposed for monitoring

#### Task 5.3: Testing Suite

**Files:** `tests/` directory

**Test Coverage:**

```python
tests/
├── unit/
│   ├── test_rag.py              # RAG retrieval accuracy
│   ├── test_jd_parser.py        # JD parsing logic
│   ├── test_matching.py         # Skill matching
│   └── test_encryption.py       # Encryption/decryption
├── integration/
│   ├── test_ollama_client.py    # Ollama connectivity
│   ├── test_job_search.py       # Job search APIs
│   └── test_chromadb.py         # Vector store operations
├── e2e/
│   ├── test_upload_flow.py      # Resume upload → storage
│   ├── test_tailor_flow.py      # Full tailoring workflow
│   └── test_search_flow.py      # Job search → save
└── fixtures/
    ├── sample_resume.pdf
    ├── sample_jd.txt
    └── mock_responses.json
```

**Critical Tests:**

```python
# tests/unit/test_rag.py
def test_resume_skills_extraction():
    """Ensure RAG retrieves correct skills."""
    resume = load_fixture("sample_resume.pdf")
    rag = RAGPipeline()
    rag.add_document(resume)

    results = rag.query("What Python frameworks does candidate know?")
    assert "FastAPI" in results
    assert "Django" in results

# tests/integration/test_tailor_flow.py
@pytest.mark.asyncio
async def test_tailoring_no_hallucinations():
    """Ensure tailored resume doesn't invent experience."""
    resume_id = await upload_resume("fixtures/sample_resume.pdf")
    jd = load_fixture("sample_jd.txt")

    result = await tailor_resume(resume_id, jd)

    # Verify all claims in tailored resume exist in original
    assert result["quality_score"] > 0.8
    assert "hallucinations" not in result or len(result["hallucinations"]) == 0

# tests/e2e/test_playwright_scraping.py
@pytest.mark.asyncio
async def test_greenhouse_scraping():
    """Test scraping a known Greenhouse career page."""
    jobs = await scrape_greenhouse("https://boards.greenhouse.io/example")

    assert len(jobs) > 0
    assert all("title" in job for job in jobs)
    assert all("description" in job for job in jobs)
```

**Add to requirements.txt:**

```python
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0  # Coverage reporting
respx==0.20.2  # Mock httpx requests
```

**Acceptance Criteria:**

- [ ] Unit tests for all utility functions (80%+ coverage)
- [ ] Integration tests for external services (Ollama, ChromaDB)
- [ ] E2E tests for critical workflows
- [ ] Tests run in CI/CD pipeline
- [ ] Mock external APIs (don't hit real APIs in tests)

---

## 📋 Technical Debt & Improvements

### Known Issues to Address

1. **CORS Configuration** (`app/main.py:38`)
   - Currently allows all origins (`allow_origins=["*"]`)
   - **Fix:** Specify exact origins in production

   ```python
   allow_origins=[
       "http://localhost:8000",
       settings.frontend_url  # Add to .env
   ]
   ```

2. **Missing API Versioning**
   - Current endpoints like `/api/submit` are unversioned
   - **Fix:** Migrate to `/api/v1/...` structure

3. **No Background Task Management**
   - Long-running tasks block HTTP requests
   - **Fix:** Implement FastAPI BackgroundTasks or ARQ

4. **ChromaDB Not Configured**
   - Dependency installed but not used
   - **Fix:** Initialize in startup event

5. **Static Files Not Used**
   - `static/` directory mounted but empty
   - **Decision:** Keep or remove?

6. **Templates Limited**
   - Only `index.html` exists
   - **Add:**
     - `resume_upload.html`
     - `job_search.html`
     - `tailored_results.html`
     - `applications.html`

### Future Enhancements (Post-MVP)

1. **Multi-User Support**
   - Add authentication (JWT tokens)
   - User management endpoints
   - User-specific data isolation

2. **Resume Version Comparison**
   - Side-by-side diff view
   - Track what changed in each version
   - Rollback to previous versions

3. **Application Automation** (⚠️ HIGH RISK)
   - Playwright form filling
   - CRITICAL: Requires explicit user approval
   - Only for standard ATS forms (Greenhouse, Lever)

4. **Email Integration**
   - Parse job alerts from email
   - Send notifications on application status changes
   - Weekly summary reports

5. **Analytics Dashboard**
   - Application funnel (applied → interview → offer)
   - Response rates by company, job type
   - Time-to-response analytics

6. **Cover Letter Generation**
   - Similar LangGraph workflow to resume tailoring
   - Templates for different job types

7. **Interview Preparation**
   - Generate likely interview questions based on JD
   - Suggest answers based on resume

---

## 🗂️ Updated Project Structure

### Current Structure (15 files)

```
AI_Slop_FastAPI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   └── routers/
│       └── basic.py
├── utils/
│   └── llm.py
├── templates/
│   └── index.html
├── docs/
│   ├── plan.md
│   └── backlog.md (this file)
├── .env
├── .env.example
├── .gitignore
├── environment.yml
├── README.md
└── requirements.txt
```

### Target Structure (Phase 4 Complete)

```
AI_Slop_FastAPI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── api.py          # Pydantic API models
│   │   └── db.py           # SQLModel database models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── basic.py
│   │   ├── resume.py       # Resume upload/query
│   │   ├── jobs.py         # Job search
│   │   ├── tailor.py       # Resume tailoring
│   │   └── applications.py # Application tracking
│   └── middleware/
│       ├── error_handler.py
│       ├── rate_limit.py
│       └── logging.py
├── agents/
│   ├── __init__.py
│   ├── graph.py            # LangGraph state machines
│   └── tailor_agent.py     # Tailoring logic
├── tools/
│   ├── __init__.py
│   └── job_search.py       # Job search tools
├── utils/
│   ├── __init__.py
│   ├── llm.py              # Ollama client
│   ├── langchain_llm.py    # LangChain wrapper
│   ├── rag.py              # RAG pipeline
│   ├── jd_parser.py        # JD analysis
│   ├── matching.py         # Resume-JD matching
│   └── encryption.py       # Data encryption
├── templates/
│   ├── index.html
│   ├── resume_upload.html
│   ├── job_search.html
│   ├── tailored_results.html
│   └── applications.html
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
├── data/
│   ├── chroma/             # ChromaDB storage
│   └── resumes/            # Uploaded files (encrypted)
├── docs/
│   ├── plan.md             # Original implementation plan
│   ├── backlog.md          # This file (current state)
│   └── api.md              # API documentation
├── alembic/                # Database migrations
├── .env
├── .env.example
├── .gitignore
├── environment.yml
├── README.md
├── requirements.txt
└── pyproject.toml          # Tool configuration
```

---

## 🔧 Environment Configuration Updates

### Additional .env Variables Needed

```bash
# Current .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b
DEBUG=True
PORT=8000

# Add these:

# Database
DATABASE_URL=sqlite:///./data/jobs.db
# For production: postgresql://user:pass@localhost/jobs

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=resumes

# Encryption
ENCRYPTION_KEY=<generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">

# Job Search APIs
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_TAILOR=5/minute
RATE_LIMIT_SEARCH=10/minute

# Background Tasks
TASK_TIMEOUT_SECONDS=300  # 5 minutes max for tailoring

# Frontend (if separated later)
FRONTEND_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:8000,http://localhost:3000

# Monitoring
ENABLE_METRICS=True
LOG_LEVEL=INFO
```

---

## 📦 Updated Dependencies

### Add to requirements.txt

```python
# Current core dependencies (keep these):
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0
python-multipart==0.0.6
jinja2==3.1.2
httpx==0.26.0
python-dotenv==1.0.0

# LangGraph (replacing CrewAI)
langgraph==0.0.25
langchain==0.1.0
langchain-community==0.0.13
langchain-ollama==0.0.1

# Vector Store & RAG
chromadb==0.4.22
# Remove: faiss-cpu==1.7.4 (not needed with ChromaDB)

# Document Processing
pypdf2==3.0.1
python-docx==1.1.0
docx2txt==0.8

# Web Scraping
playwright==1.41.0
requests==2.31.0

# Database
sqlmodel==0.0.14
alembic==1.13.1

# Security
cryptography==41.0.7
slowapi==0.1.9

# Observability
structlog==23.2.0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
respx==0.20.2

# Data Processing
pandas==2.1.4

# Optional: Background Tasks (if FastAPI BackgroundTasks insufficient)
# arq==0.25.0  # Requires Redis
```

### Remove from requirements.txt

```python
# Remove (not using):
crewai==0.1.0           # Using LangGraph instead
streamlit==1.30.0       # Using Jinja2 templates
faiss-cpu==1.7.4        # ChromaDB handles this
```

---

## 🎯 Success Metrics

### MVP Definition (Phase 1-3 Complete)

**User can:**

1. Upload resume (PDF/DOCX) ✓
2. Search for jobs (at least 2 sources) ✓
3. Get tailored resume for a specific job ✓
4. Review tailored resume before applying ✓
5. See match score for each job ✓

**System can:**

1. Process resume in < 10 seconds ✓
2. Search jobs in < 5 seconds ✓
3. Tailor resume in < 60 seconds ✓
4. Achieve 0% hallucination rate (quality check > 0.8) ✓
5. Handle 5 concurrent users ✓

### Phase Completion Criteria

**Phase 1 (RAG) Complete When:**

- [ ] Can upload and embed 10-page resume
- [ ] Query response time < 1 second
- [ ] Retrieval accuracy > 90% (manual validation)
- [ ] ChromaDB persists across restarts

**Phase 2 (Job Search) Complete When:**

- [ ] Can search RemoteOK API
- [ ] Can search Adzuna API
- [ ] Can scrape at least 2 ATS systems
- [ ] Returns 20+ jobs per search
- [ ] Match scores correlate with manual review

**Phase 3 (Tailoring) Complete When:**

- [ ] LangGraph executes full workflow
- [ ] Tailoring completes in < 60 seconds
- [ ] Quality check catches all hallucinations (0 false negatives in 10 tests)
- [ ] Tailored resumes preferred over originals (user survey)

**Phase 4 (Database) Complete When:**

- [ ] Can track 100+ job applications
- [ ] Can store 10+ resume versions
- [ ] Database migrations work
- [ ] No data loss after restart

**Phase 5 (Production) Complete When:**

- [ ] All endpoints have rate limiting
- [ ] Resume data encrypted at rest
- [ ] 80%+ test coverage
- [ ] Detailed health checks implemented
- [ ] Structured logging in place

---

## 🚨 High-Risk Areas

### 1. Playwright Scraping Fragility

**Risk Level:** HIGH
**Impact:** Job search fails silently

**Mitigation Strategy:**

- Use semantic selectors (`role="button"`, `aria-label`)
- Build fallback selector chains
- Screenshot on failure for debugging
- Limit to standard ATS (Greenhouse, Lever)
- Add alerts when scrapers break

**Acceptance Criteria:**

- [ ] Selectors survive DOM changes (test monthly)
- [ ] Failure rate < 10% per site
- [ ] Failures logged with screenshots

### 2. LLM Hallucinations

**Risk Level:** CRITICAL
**Impact:** User applies with false claims → loses credibility

**Mitigation Strategy:**

- **Strict RAG grounding:** Only use retrieved context
- **Validation prompt:** "Only use skills explicitly mentioned"
- **Multi-step quality check:** Dedicated LLM call to verify
- **Human-in-the-loop:** User approves before applying
- **Diff view:** Show exactly what changed

**Acceptance Criteria:**

- [ ] Quality check score > 0.8 for all tailored resumes
- [ ] 0 hallucinations in 100 sample runs
- [ ] User can see source for every claim

### 3. API Rate Limits

**Risk Level:** MEDIUM
**Impact:** Job search stops working mid-month

**Mitigation Strategy:**

- **Cache aggressively:** 24h TTL for job listings
- **Quota tracking:** Warn at 80% usage
- **Graceful degradation:** Fall back to scraping when API quota exhausted
- **User awareness:** Show quota status in UI

**Acceptance Criteria:**

- [ ] Never exceed API limits
- [ ] User warned at 80% quota
- [ ] Fallback mechanisms work

### 4. ChromaDB Performance

**Risk Level:** MEDIUM
**Impact:** Slow queries → poor UX

**Mitigation Strategy:**

- **Keep collections small:** Archive old resumes
- **Metadata filtering:** Narrow search space
- **Index optimization:** Proper embedding dimensions
- **Monitoring:** Track query times

**Acceptance Criteria:**

- [ ] Query time < 1 second (p95)
- [ ] Can handle 100+ resume versions
- [ ] Collections don't bloat indefinitely

### 5. Data Privacy

**Risk Level:** HIGH
**Impact:** Legal liability, user trust lost

**Mitigation Strategy:**

- **Encrypt at rest:** All resume data encrypted
- **Local-only:** No cloud LLMs, no telemetry
- **Clear data retention:** Auto-delete after 90 days option
- **User control:** Easy export/delete

**Acceptance Criteria:**

- [ ] All PII encrypted
- [ ] No data sent to third parties
- [ ] User can delete all data
- [ ] GDPR-compliant (if applicable)

---

## 📅 Timeline Estimate

### Optimistic (Full-Time, No Blockers)

- **Phase 1 (RAG):** 2-3 days
- **Phase 2 (Job Search):** 2-3 days
- **Phase 3 (Tailoring):** 3-4 days
- **Phase 4 (Database):** 1-2 days
- **Phase 5 (Production):** 2-3 days
- **Total:** 10-15 days

### Realistic (Part-Time, Expect Issues)

- **Phase 1:** 1 week
- **Phase 2:** 1 week
- **Phase 3:** 1.5 weeks (most complex)
- **Phase 4:** 3-4 days
- **Phase 5:** 1 week
- **Total:** 5-6 weeks

### Pessimistic (Learning + Debugging)

- **Phase 1:** 2 weeks (ChromaDB setup issues)
- **Phase 2:** 2 weeks (scraper breakage)
- **Phase 3:** 3 weeks (hallucination debugging)
- **Phase 4:** 1 week
- **Phase 5:** 2 weeks (security hardening)
- **Total:** 10-12 weeks

---

## 🎓 Learning Resources

### LangGraph

- Docs: <https://langchain-ai.github.io/langgraph/>
- Tutorial: Building stateful agents
- Examples: Multi-step reasoning workflows

### ChromaDB

- Docs: <https://docs.trychroma.com/>
- Tutorial: Embeddings and retrieval
- Best practices: Collection management

### Playwright

- Docs: <https://playwright.dev/python/>
- Tutorial: Async page automation
- Ethical scraping: robots.txt, delays

### SQLModel

- Docs: <https://sqlmodel.tiangolo.com/>
- Tutorial: FastAPI + SQLModel integration
- Migrations: Alembic setup

---

## 📝 Notes & Decisions

### Architecture Decisions

1. **Why LangGraph over CrewAI?**
   - More control over state management
   - Better debugging (visual graphs)
   - No heavyweight abstractions
   - Better async support

2. **Why ChromaDB over Faiss?**
   - Persistence out of the box
   - Better metadata filtering
   - Production-ready (Faiss is more for research)

3. **Why Jinja2 over Streamlit?**
   - Single codebase (no frontend/backend split)
   - Better control over UI
   - FastAPI native
   - Deployment simpler

4. **Why SQLModel over raw SQLAlchemy?**
   - Pydantic integration (same models for API + DB)
   - Type safety
   - Less boilerplate

5. **Why not use external LLM APIs (OpenAI, Anthropic)?**
   - Privacy: Resumes contain PII
   - Cost: High volume of API calls
   - Control: Local = no rate limits
   - Reliability: No external dependencies

### Open Questions

1. **Multi-user support?**
   - Current: Single-user (local tool)
   - Future: Add authentication if deploying

2. **Frontend framework?**
   - Current: Jinja2 templates (server-side)
   - Future: Consider React/Vue if complexity grows

3. **Deployment target?**
   - Current: Local (localhost)
   - Future: Docker container for VPS?

4. **Resume format for output?**
   - Current: Markdown → DOCX
   - Future: LaTeX for better formatting?

---

## 🔄 Changelog

### 2025-11-06

- Created backlog.md
- Defined 5-phase roadmap
- Updated tech stack (LangGraph, removed CrewAI)
- Added detailed task breakdown
- Identified high-risk areas
- Set success metrics

### Next Review

- After Phase 1 completion (RAG pipeline done)
- Re-evaluate LangGraph complexity
- Adjust timeline based on actual velocity

---

## 🎯 Current Sprint (Week 1)

**Goal:** Complete Phase 1 (RAG Foundation)

**Tasks:**

1. [ ] Implement `utils/rag.py` with ChromaDB
2. [ ] Add resume upload endpoint
3. [ ] Test PDF/DOCX parsing
4. [ ] Verify embeddings quality
5. [ ] Create sample queries test suite

**Definition of Done:**

- Can upload resume → embeddings stored in ChromaDB
- Can query "What are my skills?" → relevant response
- Embeddings persist across restarts
- Basic encryption implemented

**Next Sprint Preview:**

- Job search tool implementation
- API integration (RemoteOK, Adzuna)
- Basic scraper for one ATS

---

*This backlog is a living document. Update after each phase completion.*
