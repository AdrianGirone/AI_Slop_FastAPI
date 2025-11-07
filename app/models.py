"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ============ Base Text Prompt Models (Original Django Functionality) ============

class TextPromptRequest(BaseModel):
    """Request model for basic text prompt submission."""
    text: str = Field(..., min_length=1, description="Text prompt to send to AI")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What are the key skills for a Python developer?"
            }
        }


class TextPromptResponse(BaseModel):
    """Response model for AI text generation."""
    status: str = Field(..., description="Status of the request")
    message: str = Field(..., description="Status message")
    prompt: Optional[str] = Field(None, description="Original prompt")
    response: Optional[str] = Field(None, description="AI generated response")
    hint: Optional[str] = Field(None, description="Hint for troubleshooting")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Response received from AI",
                "prompt": "What are the key skills for a Python developer?",
                "response": "Key skills include Python programming, frameworks like Django/FastAPI..."
            }
        }


class SimpleQueryRequest(BaseModel):
    """Simple query request with just a prompt."""
    prompt: str = Field(..., description="Prompt text")


class SimpleQueryResponse(BaseModel):
    """Simple query response."""
    response: str = Field(..., description="AI response")


# ============ Job Search Models ============

class JobSearchRequest(BaseModel):
    """Request model for job search."""
    query: str = Field(..., min_length=1, description="Job search query (title, keywords)")
    location: str = Field(default="remote", description="Location or 'remote'")
    results_per_page: int = Field(default=20, ge=1, le=100, description="Number of results")


class JobListing(BaseModel):
    """Model for a single job listing."""
    title: str
    company: str
    location: Optional[str] = None
    description: str
    url: Optional[str] = None
    salary: Optional[str] = None
    posted_date: Optional[str] = None
    source: str = Field(..., description="Source of the listing (adzuna, usajobs, etc)")


class JobSearchResponse(BaseModel):
    """Response model for job search."""
    status: str
    jobs: List[JobListing]
    total_results: int
    query: str
    location: str


# ============ Resume/Document Models ============

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ResumeUploadResponse(BaseModel):
    """Response for resume upload."""
    status: str
    message: str
    file_path: Optional[str] = None
    document_id: Optional[str] = None


class TailoringRequest(BaseModel):
    """Request to tailor resume for a job description."""
    job_description: str = Field(..., description="Full job description text")
    resume_path: Optional[str] = Field(None, description="Path to resume file")
    document_id: Optional[str] = Field(None, description="ID of uploaded document")


class TailoringResponse(BaseModel):
    """Response from resume tailoring."""
    status: str
    message: str
    tailored_content: Optional[str] = None
    key_skills_matched: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    output_path: Optional[str] = None


# ============ Application Models ============

class ApplicationRequest(BaseModel):
    """Request to auto-apply to a job."""
    job_url: str = Field(..., description="URL of the job posting")
    resume_path: str = Field(..., description="Path to resume file")
    cover_letter: Optional[str] = Field(None, description="Optional cover letter text")
    user_approval: bool = Field(default=False, description="User must approve before applying")


class ApplicationResponse(BaseModel):
    """Response from job application."""
    status: str
    message: str
    job_url: str
    applied: bool
    application_id: Optional[str] = None
    timestamp: Optional[str] = None


# ============ Agent Status Models ============

class AgentStatus(str, Enum):
    """Status of agent operations."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    WAITING_APPROVAL = "waiting_approval"


class AgentTaskResponse(BaseModel):
    """Response for agent task status."""
    task_id: str
    status: AgentStatus
    message: str
    progress: Optional[float] = Field(None, ge=0, le=100)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============ RAG/Retrieval Models ============

class RetrievalRequest(BaseModel):
    """Request for RAG retrieval."""
    query: str = Field(..., description="Query to search in documents")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: Optional[str] = Field(None, description="Specific document to search")


class RetrievalResponse(BaseModel):
    """Response from RAG retrieval."""
    query: str
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    total_found: int


# ============ Health Check Models ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    model_available: bool
    version: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    message: str
    detail: Optional[str] = None
