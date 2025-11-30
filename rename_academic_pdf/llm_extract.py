"""
LLM-based metadata extraction for academic PDFs.

Uses OpenAI-compatible APIs to extract title, authors, year, and venue from PDF text.
Supports OpenAI, OpenRouter, and other compatible providers.

Environment variables:
- OPENAI_API_KEY: API key for OpenAI (default)
- OPENROUTER_API_KEY: API key for OpenRouter (used when model starts with provider prefix)

Model naming:
- OpenAI models: gpt-4.1-mini, gpt-4o, etc.
- OpenRouter models: Use provider/model format, e.g., anthropic/claude-3-haiku
"""

import json
import os
import re
from typing import Dict, Optional, Tuple

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_api_config(model: str) -> Tuple[str, str, Optional[str]]:
    """
    Determine API configuration based on model name.

    Args:
        model: Model name (e.g., "gpt-4.1-mini" or "anthropic/claude-3-haiku")

    Returns:
        Tuple of (api_key, model_name, base_url)
        - base_url is None for OpenAI (uses default)

    Raises:
        ValueError: If required API key is not set
    """
    # Check if model uses OpenRouter format (contains "/")
    if "/" in model:
        # OpenRouter model format: provider/model-name
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for OpenRouter models. "
                f"Model '{model}' appears to be an OpenRouter model (contains '/')."
            )
        return api_key, model, OPENROUTER_BASE_URL
    else:
        # Standard OpenAI model
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for LLM extraction"
            )
        return api_key, model, None


def extract_text_from_pdf(pdf_path: str, max_pages: int = 3) -> str:
    """
    Extract text from the first N pages of a PDF.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: 3)

    Returns:
        Extracted text as a string
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError(
            "pdfplumber is required for LLM extraction. Install with: pip install pdfplumber"
        )

    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    return "\n\n".join(text_parts)


def query_llm_for_metadata(
    text: str, model: str = "gpt-4.1-mini"
) -> Optional[Dict[str, any]]:
    """
    Use OpenAI-compatible API to extract metadata from PDF text.

    Args:
        text: Extracted text from PDF
        model: Model to use. For OpenAI: gpt-4.1-mini, gpt-4o, etc.
               For OpenRouter: use provider/model format (e.g., anthropic/claude-3-haiku)

    Returns:
        Dictionary with extracted metadata or None on failure
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai package is required for LLM extraction. Install with: pip install openai"
        )

    # Get API configuration based on model name
    api_key, model_name, base_url = get_api_config(model)

    # Create client with appropriate configuration
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)

    system_prompt = """You are an expert at extracting metadata from academic papers.
Try your best to extract the following information from the provided text and return it as JSON. If the information is not present, return empty string for that field.:

{
    "title": "Full title of the paper",
    "authors": ["First Author", "Second Author", ...],
    "year": 2024,
    "venue": "Journal or Conference Name",
    "doi": "10.xxxx/xxxxx or null if not found"
}

Rules:
- For authors, extract full names (First Last format), not just last names
- For year, use the publication year if available, otherwise use the most recent year mentioned
- For venue, use the full journal/conference name, not abbreviations
- If DOI is present in the text, extract it
- If you cannot determine a field with confidence, use null
- Return ONLY valid JSON, no other text"""

    user_prompt = f"""Extract metadata from this academic paper:

{text[:12000]}"""  # Limit text to ~12K chars to stay within token limits

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            content = json_match.group(1)

        metadata = json.loads(content)

        # Normalize the response
        result = {
            "source": "llm",
            "title": metadata.get("title"),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "doi": metadata.get("doi"),
        }

        # Handle authors
        authors = metadata.get("authors", [])
        if authors:
            # Extract last names for the authors field
            last_names = []
            for author in authors:
                if isinstance(author, str):
                    # Handle "First Last" or "Last, First" formats
                    if "," in author:
                        last_names.append(author.split(",")[0].strip())
                    else:
                        parts = author.strip().split()
                        if parts:
                            last_names.append(parts[-1])
            result["authors"] = last_names

        return result

    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse LLM response as JSON: {e}")
        return None
    except openai.APIError as e:
        print(f"  Warning: OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"  Warning: LLM extraction failed: {e}")
        return None


def extract_metadata_with_llm(
    pdf_path: str, max_pages: int = 3, model: str = "gpt-4.1-mini"
) -> Optional[Dict[str, any]]:
    """
    Extract metadata from a PDF using LLM.

    This is the main entry point for LLM-based extraction.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Number of pages to analyze (default: 3)
        model: OpenAI model to use (default: gpt-4.1-mini)

    Returns:
        Dictionary with extracted metadata or None on failure

    Example:
        >>> metadata = extract_metadata_with_llm("paper.pdf")
        >>> print(metadata)
        {
            'source': 'llm',
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani', 'Shazeer', 'Parmar', ...],
            'year': 2017,
            'venue': 'Advances in Neural Information Processing Systems',
            'doi': None
        }
    """
    print(f"  Extracting text from first {max_pages} pages...")
    text = extract_text_from_pdf(pdf_path, max_pages=max_pages)

    if not text or len(text.strip()) < 100:
        print("  Warning: Could not extract sufficient text from PDF")
        return None

    print(f"  Querying LLM ({model})...")
    return query_llm_for_metadata(text, model=model)
