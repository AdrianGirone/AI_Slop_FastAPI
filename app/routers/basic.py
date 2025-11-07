"""
Basic text prompt router - replicates original Django functionality.
Handles simple text-to-AI prompt submission and response.
"""
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import logging

from app.models import (
    TextPromptRequest,
    TextPromptResponse,
    SimpleQueryRequest,
    SimpleQueryResponse
)
from utils.llm import ollama_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["basic"])


@router.post("/submit", response_model=TextPromptResponse)
async def submit_text(request: TextPromptRequest):
    """
    Handle text submission and send to Ollama for AI response.
    Replaces Django's submit_text view with async support.

    Args:
        request: Text prompt request

    Returns:
        AI response with status
    """
    if not request.text:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "No text provided"
            }
        )

    try:
        # Send to Ollama
        response = await ollama_client.query(prompt=request.text)

        return TextPromptResponse(
            status="success",
            message="Response received from AI",
            prompt=request.text,
            response=response
        )

    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error: {str(e)}",
                "hint": "Make sure Ollama is running (ollama serve) and the model is installed (ollama pull llama3.1:70b)"
            }
        )


@router.post("/submit-form", response_model=TextPromptResponse)
async def submit_text_form(text: str = Form(...)):
    """
    Form-based text submission (for compatibility with HTML form posts).
    Accepts application/x-www-form-urlencoded or multipart/form-data.

    Args:
        text: Text from form field

    Returns:
        AI response with status
    """
    request = TextPromptRequest(text=text)
    return await submit_text(request)


@router.get("/ask", response_model=SimpleQueryResponse)
async def ai_response(prompt: str = "Say hello!"):
    """
    Simple GET-based query endpoint.
    Replaces Django's ai_response view.

    Args:
        prompt: Query parameter with prompt text

    Returns:
        Simple response object
    """
    try:
        result = await ollama_client.query(prompt=prompt)
        return SimpleQueryResponse(response=result)
    except Exception as e:
        logger.error(f"Error in ai_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=SimpleQueryResponse)
async def query_ai(request: SimpleQueryRequest):
    """
    POST endpoint for simple queries.

    Args:
        request: Simple query request

    Returns:
        AI response
    """
    try:
        result = await ollama_client.query(prompt=request.prompt)
        return SimpleQueryResponse(response=result)
    except Exception as e:
        logger.error(f"Error in query_ai: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
