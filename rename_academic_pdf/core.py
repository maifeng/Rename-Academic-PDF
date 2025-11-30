#!/usr/bin/env python3
"""
Rename Academic PDF - Core Module

Automatically renames academic PDFs using identifier-based API queries.
Uses CrossRef, DataCite, arXiv, Semantic Scholar, PubMed, OpenAlex, and DBLP APIs.

This module provides functions for extracting metadata from academic PDFs
and generating standardized filenames based on author, year, title, and journal.

Features:
- Multi-author support: Concatenates all authors when ≤5, uses "FirstAuthorEtAl" when >5
- Customizable separators: Use "-" (default) or "_" between filename components
- Custom journal abbreviations: Support for user-defined abbreviation files
- Multiple metadata sources: Cascading API queries across 7+ academic databases
- Flexible filename formats: Multiple presets and custom template support

Custom Journal Abbreviations:
The package supports custom journal abbreviation files in JSON format.
Search order for abbreviation files:
1. Command-line argument: --journal-abbrev-file path/to/file.json
2. User's home directory: ~/.rename-academic-pdf/journal_abbreviations.json
3. Default bundled file: <package>/data/journal_abbreviations.json

Custom abbreviation file format:
{
    "comment": "Optional description",
    "abbreviations": {
        "Full Journal Name": "ABBREV",
        "Another Journal": "AJ"
    }
}

Author: Feng Mai <maifeng@gmail.com>
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

import pdfplumber
import requests
from pypdf import PdfReader

# API Configuration
DOI_ORG_API = "https://doi.org/{doi}"
CROSSREF_API = "https://api.crossref.org/works/{doi}"
DATACITE_API = "https://api.datacite.org/dois/{doi}"
ARXIV_API = "http://export.arxiv.org/api/query?id_list={arxiv_id}"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/{identifier}"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
OPENALEX_API = "https://api.openalex.org/works"
DBLP_API = "https://dblp.org/search/publ/api"

# Identifier patterns
IDENTIFIER_PATTERNS = {
    "doi": r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    "arxiv": r"arXiv[:\s]*(\d{4}\.\d{4,5})",  # Matches "arXiv:1234.5678" or "arXiv 1234.5678"
    "arxiv_url": r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",  # Matches URLs
    "pmid": r"PMID:\s*(\d{7,8})",
    "pmcid": r"PMC\d{7}",
    "ssrn": r"ssrn\.com/abstract=(\d{6,8})",  # SSRN URL pattern
    "ssrn_id": r"SSRN[:\s]*(\d{6,8})",  # "SSRN: 1234567" or "SSRN 1234567"
}

# Filename patterns (for arXiv PDFs named like "2305.17808v4.pdf")
FILENAME_PATTERNS = {
    "arxiv": r"^(\d{4}\.\d{4,5})(v\d+)?\.pdf$",
    "ssrn": r"^[Ss][Ss][Rr][Nn][-_]?(?:id)?(\d{6,8})\.pdf$",  # ssrn-1234567.pdf, SSRN-id1234567.pdf, etc.
}

# Request timeout and retry settings
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2

# Filename format templates
FORMAT_TEMPLATES = {
    "default": "{author}{year}-{title}-{journal}",
    "compact": "{author}{year}-{title}",
    "full": "{author}-{year}-{title}-{journal}",
    "minimal": "{author}{year}",
    "year_first": "{year}-{author}-{title}",
    "journal_first": "{journal}-{author}{year}-{title}",
}


def load_config() -> Dict[str, any]:
    """
    Load user configuration from ~/.rename-academic-pdf/config.json.

    The config file can set default values for:
    - format: Format preset name (e.g., "compact", "minimal")
    - format_string: Custom format string (e.g., "{author}{year}-{title}")
    - separator: Separator character ("-" or "_")
    - first_author_only: Boolean, use only first author
    - llm: Boolean, enable LLM fallback
    - llm_model: LLM model name (e.g., "gpt-4.1-mini")
    - max_title_length: Maximum title length in filename (default: 80)

    Example config.json:
    {
        "format": "compact",
        "separator": "_",
        "first_author_only": false,
        "llm": true,
        "llm_model": "gpt-4.1-mini",
        "max_title_length": 100
    }

    Returns:
        Dictionary with configuration options. Empty dict if no config file.
    """
    config_path = Path.home() / ".rename-academic-pdf" / "config.json"

    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"Loaded config from {config_path}")
            return config
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


# Load journal abbreviations from external JSON file
def load_journal_abbreviations(custom_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load journal abbreviations with support for custom files.

    Search order for journal abbreviations JSON file:
    1. Custom path (if provided via custom_path parameter)
    2. User's home directory: ~/.rename-academic-pdf/journal_abbreviations.json
    3. Default bundled file: <package>/data/journal_abbreviations.json

    The JSON file should have the following structure:
    {
        "comment": "Optional description",
        "abbreviations": {
            "Full Journal Name": "ABBREV",
            "Another Journal": "AJ"
        }
    }

    Args:
        custom_path: Optional path to custom journal abbreviations JSON file.

    Returns:
        Dictionary mapping full journal names to abbreviations.
        Returns empty dict if no valid file found.
    """
    search_paths = []

    # 1. Custom path from parameter (usually from CLI argument)
    if custom_path:
        search_paths.append((Path(custom_path), "custom path"))

    # 2. User's home directory
    home_path = Path.home() / ".rename-academic-pdf" / "journal_abbreviations.json"
    search_paths.append((home_path, "user home directory"))

    # 3. Default bundled file
    default_path = Path(__file__).parent / "data" / "journal_abbreviations.json"
    search_paths.append((default_path, "default bundled file"))

    for json_path, source in search_paths:
        try:
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    abbrevs = data.get("abbreviations", {})
                    if abbrevs:  # Only use if non-empty
                        print(
                            f"Loaded {len(abbrevs)} journal abbreviations from {source}: {json_path}"
                        )
                        return abbrevs
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_path}: {e}")
            continue
        except Exception as e:
            # Catch other errors like permission issues
            continue

    print("Warning: No valid journal abbreviations file found.")
    print("Using auto-generated abbreviations only.")
    return {}


# Load journal abbreviations on module import
JOURNAL_ABBREVIATIONS = load_journal_abbreviations()


def normalize_venue_name(name: str) -> str:
    """
    Normalize venue name for matching.

    Removes punctuation, collapses whitespace, and lowercases.
    """
    # Remove common punctuation and special chars
    normalized = re.sub(r'[.,;:!?\'"()\[\]{}]', "", name)
    # Replace & with 'and' for consistency
    normalized = normalized.replace("&", " and ")
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def abbreviate_journal(
    venue: str,
    abbreviations: Optional[Dict[str, str]] = None,
    venue_full: Optional[str] = None,
) -> str:
    """
    Abbreviate journal/venue name if possible.

    Args:
        venue: Journal or venue name (may be short title from API).
        abbreviations: Optional custom journal abbreviations dict.
            If None, uses the default JOURNAL_ABBREVIATIONS loaded at module import.
        venue_full: Optional full journal name for abbreviation lookup.
            Used when venue is already a short title from API.

    Returns:
        Abbreviated name if found, otherwise original or generated abbreviation.
    """
    if not venue:
        return ""

    # Use provided abbreviations or fall back to default
    abbrev_dict = abbreviations if abbreviations is not None else JOURNAL_ABBREVIATIONS

    # Clean up HTML entities and extra whitespace
    venue_clean = venue.replace("&amp;", "&").strip()
    venue_normalized = normalize_venue_name(venue_clean)

    # Check curated abbreviations with normalized matching
    for full_name, abbrev in abbrev_dict.items():
        if venue_normalized == normalize_venue_name(full_name):
            return abbrev

    # If venue_full provided, try to look up abbreviation using full name
    if venue_full:
        venue_full_normalized = normalize_venue_name(
            venue_full.replace("&amp;", "&").strip()
        )
        for full_name, abbrev in abbrev_dict.items():
            if venue_full_normalized == normalize_venue_name(full_name):
                return abbrev
        # If full name found in our dict, use that abbreviation
        # Otherwise, the API's short title is probably good, keep it

    # If venue is already short (<=6 chars and all caps), keep it
    if len(venue_clean) <= 6 and venue_clean.isupper():
        return venue_clean

    # Generate abbreviation from capital letters and first letters of words
    # Strategy: Take capital letters or first letter of each major word
    words = venue_clean.split()

    # If it's already mostly abbreviations (e.g., "IEEE TPAMI"), keep it
    if all(len(word) <= 4 and word[0].isupper() for word in words if word):
        return " ".join(words)

    # Generate abbreviation: Take first letter of each major word
    # Skip common small words like "of", "the", "and", "for", "in"
    skip_words = {"of", "the", "and", "for", "in", "on", "a", "an", "with", "to"}
    abbrev_parts = []

    for word in words:
        if word.lower() not in skip_words and word:
            # If word starts with capital, take it
            if word[0].isupper():
                abbrev_parts.append(word[0])

    generated_abbrev = "".join(abbrev_parts)

    # If generated abbreviation is reasonable (2-8 chars), use it
    if 2 <= len(generated_abbrev) <= 8:
        return generated_abbrev.upper()

    # Otherwise, return first 20 chars of original
    return venue_clean[:20]


def clean_filename(
    text: str, max_length: int = 100, truncate_words: bool = True
) -> str:
    """
    Clean text to be filesystem-safe.

    Args:
        text: Text to clean
        max_length: Maximum length of output
        truncate_words: If True, truncate at word boundary (don't cut mid-word)

    Returns:
        Cleaned filename-safe string
    """
    if not text:
        return ""

    # Remove or replace problematic characters
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    # Replace multiple spaces/underscores with single space
    text = re.sub(r"[\s_]+", " ", text)
    # Remove leading/trailing spaces and dots
    text = text.strip(" .")

    # Truncate to max_length
    if len(text) <= max_length:
        return text

    if truncate_words:
        # Truncate at word boundary - find last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            # Only use word boundary if it's not too far back
            return truncated[:last_space].rstrip()
        # Fall back to hard truncation if no good word boundary
        return truncated

    return text[:max_length]


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 3) -> str:
    """
    Extract text from first few pages of PDF.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract

    Returns:
        Extracted text from PDF
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for page in pdf.pages[:max_pages]:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
    except Exception as e:
        print(f"  Warning: Error extracting PDF text: {e}")
        return ""


def extract_pdf_metadata(pdf_path: Path) -> Dict[str, str]:
    """
    Extract embedded metadata from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary of metadata fields
    """
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata

        result = {}
        if metadata:
            # Check for DOI in metadata
            for key in ["/doi", "/DOI", "/Subject"]:
                value = metadata.get(key, "")
                if value and "10." in value:
                    # Extract DOI from metadata
                    doi_match = re.search(IDENTIFIER_PATTERNS["doi"], value)
                    if doi_match:
                        result["doi"] = doi_match.group(0)
                        break

        return result
    except Exception as e:
        print(f"  Warning: Error extracting PDF metadata: {e}")
        return {}


def extract_identifiers(pdf_path: Path) -> Dict[str, str]:
    """
    Extract all identifiers (DOI, arXiv, PMID, etc.) from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary mapping identifier type to value
    """
    identifiers = {}

    # Check embedded metadata first
    pdf_metadata = extract_pdf_metadata(pdf_path)
    if pdf_metadata.get("doi"):
        identifiers["doi"] = pdf_metadata["doi"]

    # Check filename for arXiv ID (common for downloaded arXiv PDFs)
    filename = pdf_path.name
    for id_type, pattern in FILENAME_PATTERNS.items():
        match = re.match(pattern, filename)
        if match:
            identifiers[id_type] = match.group(1)

    # Extract text from first few pages
    text = extract_text_from_pdf(pdf_path, max_pages=3)

    # Extract all identifier types from text
    # Some PDFs have reversed text, so we search both forward and backward
    texts_to_search = [text, text[::-1]]  # Forward and reversed

    for id_type, pattern in IDENTIFIER_PATTERNS.items():
        # Map variant patterns to canonical type
        if id_type == "arxiv_url":
            normalized_type = "arxiv"
        elif id_type == "ssrn_id":
            normalized_type = "ssrn"
        else:
            normalized_type = id_type

        if normalized_type not in identifiers:  # Don't override metadata or filename
            for search_text in texts_to_search:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    # Use group(1) if capturing group exists, else group(0)
                    identifiers[normalized_type] = (
                        match.group(1) if match.groups() else match.group(0)
                    )
                    # Clean up DOI
                    if normalized_type == "doi":
                        identifiers[normalized_type] = identifiers[
                            normalized_type
                        ].strip(".,;")
                    break  # Found it, no need to search reversed

    return identifiers


def query_doi_org(doi: str) -> Optional[Dict[str, any]]:
    """
    Query DOI.org metadata API for DOI resolution.

    This is the authoritative DOI resolver that returns Citeproc JSON.
    Should be tried first before CrossRef or DataCite.

    Args:
        doi: Digital Object Identifier

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        url = DOI_ORG_API.format(doi=quote(doi, safe=""))
        # Request Citeproc JSON format
        headers = {
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "PDFRenamer/1.0",
        }

        response = requests.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )

        if response.status_code == 200:
            data = response.json()

            # Extract authors
            authors = []
            for author in data.get("author", [])[:6]:
                if "family" in author:
                    authors.append(author["family"])
                elif "name" in author:  # For organizations
                    authors.append(author["name"].split()[-1])

            # Get publication year
            year = ""
            date_parts = data.get("issued", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])

            # Get journal/venue
            venue = data.get("container-title", "") or data.get("publisher", "")

            # Get title
            title = data.get("title", "")

            if authors and title:  # Only return if we got useful data
                return {
                    "authors": authors,
                    "year": year,
                    "title": title,
                    "venue": venue,
                    "source": "doi.org",
                }
    except Exception as e:
        print(f"  DOI.org query failed: {e}")

    return None


def query_crossref(doi: str) -> Optional[Dict[str, any]]:
    """
    Query CrossRef API for DOI metadata.

    Args:
        doi: Digital Object Identifier

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        # Use polite pool with email for better rate limits
        email = os.getenv("EMAIL", "user@example.com")
        url = CROSSREF_API.format(doi=quote(doi, safe=""))
        headers = {"User-Agent": f"PDFRenamer/1.0 (mailto:{email})"}

        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})

            # Extract authors
            authors = []
            for author in message.get("author", [])[
                :6
            ]:  # Up to 6 authors (for ≤5 concatenation rule)
                if "family" in author:
                    authors.append(author["family"])

            # Get publication year
            date_parts = message.get(
                "published-print", message.get("published-online", {})
            ).get("date-parts", [[]])[0]
            year = str(date_parts[0]) if date_parts else ""

            # Get journal/venue
            # Prefer short-container-title (CrossRef's abbreviation), fall back to full title
            short_title = message.get("short-container-title", [""])[0]
            full_title = message.get("container-title", [""])[0]
            venue = short_title or full_title or message.get("publisher", "")

            # Also store full title for abbreviation lookup if short title was used
            venue_full = full_title if short_title else None

            # Get title
            title = message.get("title", [""])[0]

            return {
                "authors": authors,
                "year": year,
                "title": title,
                "venue": venue,
                "venue_full": venue_full,  # For abbreviation lookup fallback
                "source": "crossref",
            }
    except Exception as e:
        print(f"  CrossRef query failed: {e}")

    return None


def query_datacite(doi: str) -> Optional[Dict[str, any]]:
    """
    Query DataCite API for DOI metadata (datasets, conferences).

    Args:
        doi: Digital Object Identifier

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        url = DATACITE_API.format(doi=quote(doi, safe=""))
        response = requests.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json().get("data", {})
            attributes = data.get("attributes", {})

            # Extract authors
            authors = []
            for creator in attributes.get("creators", [])[:6]:
                name = creator.get("familyName") or creator.get("name", "").split()[-1]
                if name:
                    authors.append(name)

            # Get publication year
            year = str(attributes.get("publicationYear", ""))

            # Get title
            titles = attributes.get("titles", [])
            title = titles[0].get("title", "") if titles else ""

            # Get publisher as venue
            venue = attributes.get("publisher", "")

            return {
                "authors": authors,
                "year": year,
                "title": title,
                "venue": venue,
                "source": "datacite",
            }
    except Exception as e:
        print(f"  DataCite query failed: {e}")

    return None


def query_arxiv(arxiv_id: str) -> Optional[Dict[str, any]]:
    """
    Query arXiv API for preprint metadata.

    Args:
        arxiv_id: arXiv identifier (e.g., 2301.12345)

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        url = ARXIV_API.format(arxiv_id=arxiv_id)
        response = requests.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

            # Extract authors
            authors = []
            for author in entry.findall("atom:author", ns)[:6]:
                name_elem = author.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    # Get last name
                    parts = name_elem.text.strip().split()
                    authors.append(parts[-1] if parts else "")

            # Get title
            title_elem = entry.find("atom:title", ns)
            title = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None
                else ""
            )

            # Get publication date
            published_elem = entry.find("atom:published", ns)
            year = ""
            if published_elem is not None and published_elem.text:
                year = published_elem.text[:4]  # Extract year from date

            # Check for journal reference
            journal_elem = entry.find("atom:journal_ref", ns)
            venue = (
                journal_elem.text
                if journal_elem is not None and journal_elem.text
                else "arXiv"
            )

            return {
                "authors": authors,
                "year": year,
                "title": title,
                "venue": venue,
                "source": "arxiv",
            }
    except Exception as e:
        print(f"  arXiv query failed: {e}")

    return None


def query_semantic_scholar(
    identifier: str, search_type: str = "DOI"
) -> Optional[Dict[str, any]]:
    """
    Query Semantic Scholar API for paper metadata.

    Args:
        identifier: DOI, arXiv ID, or paper title
        search_type: 'DOI', 'arXiv', or 'title'

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        query_title = identifier if search_type == "title" else None

        # Format identifier based on type
        if search_type == "DOI":
            query_id = f"DOI:{identifier}"
        elif search_type == "arXiv":
            query_id = f"arXiv:{identifier}"
        else:  # title search
            # Use search endpoint for title
            search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(identifier)}&limit=1"
            response = requests.get(
                search_url, headers=headers, timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                if not papers:
                    return None
                query_id = papers[0].get("paperId")
            else:
                return None

        # Get paper details
        fields = "title,authors,year,venue"
        url = SEMANTIC_SCHOLAR_API.format(identifier=query_id)
        url += f"?fields={fields}"

        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()

            # For title searches, validate the result matches
            result_title = data.get("title", "")
            if query_title and not titles_match(query_title, result_title):
                print(f"    Title mismatch, skipping Semantic Scholar result")
                return None

            # Extract authors (last names only)
            authors = []
            for author in data.get("authors", [])[:6]:
                name = author.get("name", "")
                if name:
                    # Get last name
                    parts = name.split()
                    authors.append(parts[-1] if parts else name)

            return {
                "authors": authors,
                "year": str(data.get("year", "")),
                "title": result_title,
                "venue": data.get("venue", ""),
                "source": "semantic_scholar",
            }
    except Exception as e:
        print(f"  Semantic Scholar query failed: {e}")

    return None


def query_pubmed(pmid: str) -> Optional[Dict[str, any]]:
    """
    Query PubMed API for biomedical article metadata.

    Args:
        pmid: PubMed ID

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "json"}

        api_key = os.getenv("PUBMED_API_KEY")
        if api_key:
            params["api_key"] = api_key

        response = requests.get(PUBMED_API, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {}).get(pmid, {})

            if not result:
                return None

            # Extract authors
            authors = []
            for author in result.get("authors", [])[:6]:
                name = author.get("name", "")
                if name:
                    # PubMed format: "Last FM"
                    parts = name.split()
                    authors.append(parts[0] if parts else name)

            # Get year
            pubdate = result.get("pubdate", "")
            year = pubdate.split()[0] if pubdate else ""

            # Get title
            title = result.get("title", "")

            # Get journal
            venue = result.get("fulljournalname", result.get("source", ""))

            return {
                "authors": authors,
                "year": year,
                "title": title,
                "venue": venue,
                "source": "pubmed",
            }
    except Exception as e:
        print(f"  PubMed query failed: {e}")

    return None


def titles_match(query_title: str, result_title: str, threshold: float = 0.7) -> bool:
    """
    Check if two titles are similar enough to be considered a match.

    Uses word overlap ratio to determine similarity.

    Args:
        query_title: The title we searched for.
        result_title: The title returned by the API.
        threshold: Minimum similarity ratio (0-1) to consider a match.

    Returns:
        True if titles are similar enough, False otherwise.
    """
    if not query_title or not result_title:
        return False

    # Normalize: lowercase, remove punctuation, split into words
    def normalize(text: str) -> set:
        text = re.sub(r"[^\w\s]", "", text.lower())
        return set(text.split())

    query_words = normalize(query_title)
    result_words = normalize(result_title)

    if not query_words or not result_words:
        return False

    # Calculate Jaccard similarity (intersection over union)
    intersection = len(query_words & result_words)
    union = len(query_words | result_words)

    similarity = intersection / union if union > 0 else 0

    return similarity >= threshold


def query_openalex(query_title: str) -> Optional[Dict[str, any]]:
    """
    Query OpenAlex API for paper metadata by title.

    OpenAlex is a comprehensive open catalog of scholarly works (200M+ papers).
    Free, no API key required, excellent coverage.

    Args:
        query_title: Paper title to search for

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        # OpenAlex search by title
        params = {"filter": f"display_name.search:{query_title}", "per-page": 1}

        # Add polite pool identifier
        headers = {"User-Agent": "PDFRenamer/1.0 (mailto:user@example.com)"}

        response = requests.get(
            OPENALEX_API, params=params, headers=headers, timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return None

            work = results[0]

            # Get title and validate it matches our query
            result_title = work.get("display_name", "")
            if not titles_match(query_title, result_title):
                print(f"    Title mismatch, skipping OpenAlex result")
                return None

            # Extract authors (last names only)
            authors = []
            for authorship in work.get("authorships", [])[:6]:
                author = authorship.get("author", {})
                display_name = author.get("display_name", "")
                if display_name:
                    # Get last name
                    parts = display_name.split()
                    authors.append(parts[-1] if parts else display_name)

            # Get publication year
            year = str(work.get("publication_year", ""))

            # Get venue (journal or conference)
            venue = ""
            primary_location = work.get("primary_location")
            if primary_location and primary_location.get("source"):
                venue = primary_location["source"].get("display_name", "")

            # Fallback: try to get venue from locations list
            if not venue:
                locations = work.get("locations", [])
                for loc in locations:
                    source = loc.get("source")
                    if source and source.get("display_name"):
                        venue = source["display_name"]
                        break

            return {
                "authors": authors,
                "year": year,
                "title": result_title,
                "venue": venue,
                "source": "openalex",
            }
    except Exception as e:
        print(f"  OpenAlex query failed: {e}")

    return None


def query_dblp(query_title: str) -> Optional[Dict[str, any]]:
    """
    Query DBLP API for computer science paper metadata by title.

    DBLP is the comprehensive CS bibliography database.
    Best for computer science papers.

    Args:
        query_title: Paper title to search for

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        params = {"q": query_title, "format": "json", "h": 1}  # Return only 1 result

        response = requests.get(DBLP_API, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])

            if not hits:
                return None

            info = hits[0].get("info", {})

            # Get title and validate it matches our query
            result_title = info.get("title", "")
            if not titles_match(query_title, result_title):
                print(f"    Title mismatch, skipping DBLP result")
                return None

            # Extract authors (last names only)
            authors = []
            authors_data = info.get("authors", {}).get("author", [])
            # Handle both single author (dict) and multiple authors (list)
            if isinstance(authors_data, dict):
                authors_data = [authors_data]

            for author in authors_data[:6]:
                if isinstance(author, dict):
                    name = author.get("text", "")
                else:
                    name = str(author)

                if name:
                    # Get last name (DBLP format: "First Last" or "First M. Last")
                    parts = name.split()
                    authors.append(parts[-1] if parts else name)

            # Get publication year
            year = str(info.get("year", ""))

            # Get venue
            venue = info.get("venue", "")

            return {
                "authors": authors,
                "year": year,
                "title": result_title,
                "venue": venue,
                "source": "dblp",
            }
    except Exception as e:
        print(f"  DBLP query failed: {e}")

    return None


def extract_title_heuristic(pdf_path: Path) -> str:
    """
    Extract likely title from PDF using heuristics (fallback method).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Likely title string
    """
    text = extract_text_from_pdf(pdf_path, max_pages=1)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Look for substantial lines that could be titles
    for line in lines[:15]:
        if (
            len(line) > 20
            and not re.match(r"^\d+$", line)  # Not just numbers
            and not re.match(r"^[A-Z\s]+$", line)  # Not all caps
            and "abstract" not in line.lower()
            and "arxiv" not in line.lower()
            and "doi" not in line.lower()
        ):
            return line

    return ""


def extract_metadata_cascade(pdf_path: Path) -> Optional[Dict[str, any]]:
    """
    Extract metadata using cascade of API services (Zotero-inspired).

    Tries identifiers in priority order:
    1. DOI → DOI.org → CrossRef → DataCite → Semantic Scholar
    2. arXiv ID → arXiv API → Semantic Scholar
    3. SSRN ID → Convert to DOI (10.2139/ssrn.{id}) → DOI.org
    4. PMID → PubMed API
    5. Title-based search → Semantic Scholar → DBLP → OpenAlex

    APIs used:
    - DOI.org: Authoritative DOI resolver (Citeproc JSON)
    - CrossRef: 130M+ journal articles (DOI-based, including SSRN)
    - DataCite: Datasets, conferences (DOI-based)
    - arXiv: Preprints in STEM fields
    - PubMed: Biomedical literature
    - Semantic Scholar: 200M+ papers (uses your API key)
    - DBLP: Computer science bibliography
    - OpenAlex: 200M+ works across all fields (FREE, last resort)

    Args:
        pdf_path: Path to PDF file

    Returns:
        Metadata dictionary with authors, year, title, venue, source
    """
    print("  Extracting identifiers from PDF...")
    identifiers = extract_identifiers(pdf_path)

    if identifiers:
        print(
            f"  Found identifiers: {', '.join(f'{k.upper()}: {v}' for k, v in identifiers.items())}"
        )
    else:
        print("  No identifiers found in PDF")

    metadata = None

    # 1. Try DOI (DOI.org → CrossRef → DataCite → Semantic Scholar)
    if identifiers.get("doi"):
        print(f"  Querying DOI.org for DOI: {identifiers['doi']}")
        metadata = query_doi_org(identifiers["doi"])

        if not metadata:
            print(f"  Querying CrossRef for DOI: {identifiers['doi']}")
            metadata = query_crossref(identifiers["doi"])

        if not metadata:
            print(f"  Querying DataCite for DOI: {identifiers['doi']}")
            metadata = query_datacite(identifiers["doi"])

        # Also try Semantic Scholar with DOI as fallback
        if not metadata:
            print(f"  Querying Semantic Scholar for DOI: {identifiers['doi']}")
            metadata = query_semantic_scholar(identifiers["doi"], search_type="DOI")

    # 2. Try arXiv
    if not metadata and identifiers.get("arxiv"):
        print(f"  Querying arXiv for ID: {identifiers['arxiv']}")
        metadata = query_arxiv(identifiers["arxiv"])

        # Try Semantic Scholar with arXiv ID
        if not metadata:
            print(f"  Querying Semantic Scholar for arXiv: {identifiers['arxiv']}")
            metadata = query_semantic_scholar(identifiers["arxiv"], search_type="arXiv")

    # 3. Try SSRN (convert SSRN ID to DOI: 10.2139/ssrn.{id})
    if not metadata and identifiers.get("ssrn"):
        ssrn_doi = f"10.2139/ssrn.{identifiers['ssrn']}"
        print(f"  Converting SSRN ID to DOI: {ssrn_doi}")
        print(f"  Querying DOI.org for SSRN DOI...")
        metadata = query_doi_org(ssrn_doi)
        if not metadata:
            print(f"  Querying CrossRef for SSRN DOI...")
            metadata = query_crossref(ssrn_doi)

    # 4. Try PubMed
    if not metadata and identifiers.get("pmid"):
        print(f"  Querying PubMed for PMID: {identifiers['pmid']}")
        metadata = query_pubmed(identifiers["pmid"])

    # 5. Fallback: Try title-based searches across multiple APIs
    if not metadata:
        print("  No metadata from identifiers, trying title-based search...")
        title = extract_title_heuristic(pdf_path)

        if title:
            print(f"  Extracted title: {title[:60]}...")

            # Try multiple title-based search APIs
            # Semantic Scholar: Good coverage, strong CS/AI focus
            print(f"  Querying Semantic Scholar...")
            metadata = query_semantic_scholar(title, search_type="title")

            # DBLP: Best for computer science papers
            if not metadata:
                print(f"  Querying DBLP (Computer Science)...")
                metadata = query_dblp(title)

            # OpenAlex: Broadest coverage (200M+ papers), last resort
            if not metadata:
                print(f"  Querying OpenAlex...")
                metadata = query_openalex(title)

    if metadata:
        print(
            f"  ✓ Metadata retrieved from {metadata.get('source', 'unknown').upper()}"
        )
    else:
        print("  ✗ Could not retrieve metadata from any source")

    return metadata


def format_filename_from_metadata(
    metadata: Dict[str, any],
    format_string: str = None,
    first_author_only: bool = False,
    separator: str = "-",
    abbreviations: Optional[Dict[str, str]] = None,
    max_title_length: int = 80,
) -> str:
    """
    Format filename from metadata using customizable template.

    Args:
        metadata: Dictionary with authors, year, title, venue fields.
        format_string: Format template string (e.g., "{author}{year}-{title}").
            If None, uses default template.
        first_author_only: If True, use only first author (default: False).
        separator: Separator character, either "-" or "_" (default: "-").
        abbreviations: Optional custom journal abbreviations dict.
            If None, uses default abbreviations.
        max_title_length: Maximum length for title in filename (default: 80).
            Truncates at word boundary to avoid cutting mid-word.

    Returns:
        Formatted filename string.

    Available template variables:
        {author}  - Author name(s): all authors concatenated if <=5, else FirstAuthorEtAl
        {year}    - Publication year
        {title}   - Paper title
        {journal} - Journal/venue abbreviation
    """
    # Use default format if not specified
    if format_string is None:
        format_string = FORMAT_TEMPLATES["default"]

    # Extract and clean components
    authors = metadata.get("authors", [])

    # Handle author names based on preferences
    if first_author_only:
        # Use only first author
        author = clean_filename(authors[0], max_length=30) if authors else ""
    elif len(authors) <= 5:
        # Concatenate all authors (<= 5 authors)
        author = "".join([clean_filename(a, max_length=20) for a in authors])
    else:
        # Use FirstAuthorEtAl for papers with > 5 authors
        first = clean_filename(authors[0], max_length=20) if authors else ""
        author = f"{first}EtAl"

    year = metadata.get("year", "")
    title = clean_filename(metadata.get("title", ""), max_length=max_title_length)

    # Abbreviate and clean journal
    venue = metadata.get("venue", "")
    venue_full = metadata.get("venue_full")  # Full title for abbreviation lookup
    journal = ""
    if venue:
        venue_abbrev = abbreviate_journal(
            venue, abbreviations=abbreviations, venue_full=venue_full
        )
        journal = clean_filename(venue_abbrev, max_length=20)

    # Build components dict
    components = {
        "author": author,
        "year": str(year) if year else "",
        "title": title,
        "journal": journal,
    }

    # Replace template variables
    filename = format_string
    for key, value in components.items():
        placeholder = f"{{{key}}}"
        if placeholder in filename:
            filename = filename.replace(placeholder, value if value else "")

    # Replace separator if underscore is specified
    if separator == "_":
        filename = filename.replace("-", "_")

    # Clean up multiple separators/spaces
    if separator == "_":
        filename = re.sub(r"_+", "_", filename)  # Multiple underscores to single
    else:
        filename = re.sub(r"-+", "-", filename)  # Multiple dashes to single
    filename = re.sub(r"\s+", " ", filename)  # Multiple spaces to single
    filename = filename.strip(separator).strip()  # Remove leading/trailing separators

    if filename:
        return filename + ".pdf"
    else:
        return None


def rename_pdf(
    pdf_path: Path,
    dry_run: bool = False,
    format_string: str = None,
    first_author_only: bool = False,
    separator: str = "-",
    abbreviations: Optional[Dict[str, str]] = None,
    use_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
    max_title_length: int = 80,
) -> bool:
    """
    Rename PDF file based on metadata from API services.

    Args:
        pdf_path: Path to the PDF file.
        dry_run: If True, only show what would be done without renaming.
        format_string: Custom format template (e.g., "{author}{year}-{title}").
        first_author_only: If True, use only first author (default: False).
        separator: Separator character, "-" or "_" (default: "-").
        abbreviations: Optional custom journal abbreviations dict.
            If None, uses default abbreviations.
        use_llm: If True, use LLM to extract metadata from PDF text.
            Requires OPENAI_API_KEY environment variable.
        llm_model: LLM model to use for LLM extraction (default: gpt-4.1-mini).
        max_title_length: Maximum length for title in filename (default: 80).
            Truncates at word boundary to avoid cutting mid-word.
        llm_model: OpenAI model to use for LLM extraction (default: gpt-4.1-mini).

    Returns:
        True if successful, False otherwise.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' not found")
        return False

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"Error: File '{pdf_path}' is not a PDF")
        return False

    print(f"\nProcessing: {pdf_path.name}")
    print("=" * 70)

    # Extract metadata using cascade approach, with optional LLM fallback
    metadata = extract_metadata_cascade(pdf_path)

    # If API cascade failed and LLM is enabled, try LLM as last resort
    if not metadata and use_llm:
        try:
            from .llm_extract import extract_metadata_with_llm

            print("\nAPI extraction failed. Trying LLM as fallback...")
            metadata = extract_metadata_with_llm(pdf_path, model=llm_model)
        except ImportError as e:
            print(f"\nWarning: LLM extraction not available: {e}")
        except Exception as e:
            print(f"\nWarning: LLM extraction failed: {e}")

    if not metadata:
        print("\n✗ Could not extract metadata from any source")
        return False

    # Display extracted metadata
    print("\nExtracted metadata:")
    print(f"  Authors: {', '.join(metadata.get('authors', ['N/A']))}")
    print(f"  Year: {metadata.get('year', 'N/A')}")
    print(f"  Title: {metadata.get('title', 'N/A')}")
    print(f"  Venue: {metadata.get('venue', 'N/A')}")

    # Generate new filename with custom format
    new_filename = format_filename_from_metadata(
        metadata,
        format_string=format_string,
        first_author_only=first_author_only,
        separator=separator,
        abbreviations=abbreviations,
        max_title_length=max_title_length,
    )

    if not new_filename:
        print("\n✗ Could not generate filename from metadata")
        return False

    new_path = pdf_path.parent / new_filename

    # Check if file already exists
    if new_path.exists() and new_path != pdf_path:
        print(f"\n⚠ Warning: File '{new_filename}' already exists")
        response = input("Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Rename cancelled")
            return False

    print(f"\nNew filename: {new_filename}")

    if dry_run:
        print("\n[DRY RUN] Would rename:")
        print(f"  From: {pdf_path}")
        print(f"  To:   {new_path}")
        return True

    # Perform rename
    try:
        pdf_path.rename(new_path)
        print(f"\n✓ Successfully renamed!")
        return True
    except Exception as e:
        print(f"\n✗ Error renaming file: {e}")
        return False


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python rename_pdf.py paper.pdf")
        print("  python rename_pdf.py paper.pdf --dry-run")
        print("  python rename_pdf.py *.pdf --format compact")
        print(
            "  python rename_pdf.py paper.pdf --format-string '{year}-{author}-{title}'"
        )
        print(
            "  python rename_pdf.py paper.pdf --llm  # Use LLM (requires OPENAI_API_KEY)"
        )
        print("\nAvailable format presets:")
        for name, template in FORMAT_TEMPLATES.items():
            print(f"  {name:15s} {template}")
        sys.exit(1)

    # Load config file defaults
    config = load_config()

    # Parse arguments (CLI args override config file)
    args = sys.argv[1:]
    dry_run = "--dry-run" in args or "-n" in args
    first_author_only = "--first-author-only" in args or config.get(
        "first_author_only", False
    )

    # Parse format option (CLI overrides config)
    format_string = None
    if "--format" in args:
        idx = args.index("--format")
        if idx + 1 < len(args):
            format_preset = args[idx + 1]
            if format_preset in FORMAT_TEMPLATES:
                format_string = FORMAT_TEMPLATES[format_preset]
            else:
                print(f"Error: Unknown format preset '{format_preset}'")
                print(f"Available presets: {', '.join(FORMAT_TEMPLATES.keys())}")
                sys.exit(1)
    elif config.get("format"):
        # Use config file format preset
        if config["format"] in FORMAT_TEMPLATES:
            format_string = FORMAT_TEMPLATES[config["format"]]
        else:
            print(
                f"Warning: Unknown format preset '{config['format']}' in config, ignoring"
            )

    # Parse custom format string (CLI overrides config)
    if "--format-string" in args:
        idx = args.index("--format-string")
        if idx + 1 < len(args):
            format_string = args[idx + 1]
        else:
            print("Error: --format-string requires a template argument")
            sys.exit(1)
    elif not format_string and config.get("format_string"):
        # Use config file format string (only if no format preset was set)
        format_string = config["format_string"]

    # Parse separator option (CLI overrides config)
    separator = config.get("separator", "-")  # default from config or "-"
    if "--separator" in args:
        idx = args.index("--separator")
        if idx + 1 < len(args):
            sep_value = args[idx + 1]
            if sep_value in ["-", "_"]:
                separator = sep_value
            else:
                print(f"Error: --separator must be '-' or '_', got '{sep_value}'")
                sys.exit(1)
        else:
            print("Error: --separator requires an argument ('-' or '_')")
            sys.exit(1)

    # Parse journal abbreviations file option
    journal_abbrev_path = None
    if "--journal-abbrev-file" in args:
        idx = args.index("--journal-abbrev-file")
        if idx + 1 < len(args):
            journal_abbrev_path = args[idx + 1]
        else:
            print("Error: --journal-abbrev-file requires a file path argument")
            sys.exit(1)

    # Parse LLM option (CLI overrides config)
    use_llm = "--llm" in args or config.get("llm", False)
    llm_model = config.get("llm_model", "gpt-4.1-mini")  # default from config
    if "--llm-model" in args:
        idx = args.index("--llm-model")
        if idx + 1 < len(args):
            llm_model = args[idx + 1]
        else:
            print("Error: --llm-model requires a model name argument")
            sys.exit(1)

    # Parse max_title_length option (CLI overrides config)
    max_title_length = config.get("max_title_length", 80)  # default 80
    if "--max-title-length" in args:
        idx = args.index("--max-title-length")
        if idx + 1 < len(args):
            try:
                max_title_length = int(args[idx + 1])
                if max_title_length < 10:
                    print("Error: --max-title-length must be at least 10")
                    sys.exit(1)
            except ValueError:
                print(
                    f"Error: --max-title-length must be an integer, got '{args[idx + 1]}'"
                )
                sys.exit(1)
        else:
            print("Error: --max-title-length requires a number argument")
            sys.exit(1)

    # Load custom journal abbreviations if specified
    abbreviations = None
    if journal_abbrev_path:
        abbreviations = load_journal_abbreviations(custom_path=journal_abbrev_path)
        if not abbreviations:
            print(f"Warning: Could not load abbreviations from {journal_abbrev_path}")
            print("Continuing with default abbreviations...")

    # Get all PDF file arguments (excluding flags and their values)
    skip_next = False
    pdf_files = []
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in [
            "--format",
            "--format-string",
            "--separator",
            "--journal-abbrev-file",
            "--llm-model",
            "--max-title-length",
        ]:
            skip_next = True
            continue
        if not arg.startswith("-"):
            pdf_files.append(arg)

    if not pdf_files:
        print("Error: No PDF files specified")
        sys.exit(1)

    # Show format being used
    if format_string:
        print(f"Using format: {format_string}\n")

    # Process each PDF file
    results = []
    for i, pdf_path in enumerate(pdf_files):
        if len(pdf_files) > 1:
            print(f"\n{'='*70}")
            print(f"File {i+1}/{len(pdf_files)}")
            print(f"{'='*70}")

        success = rename_pdf(
            pdf_path,
            dry_run=dry_run,
            format_string=format_string,
            first_author_only=first_author_only,
            separator=separator,
            abbreviations=abbreviations,
            use_llm=use_llm,
            llm_model=llm_model,
            max_title_length=max_title_length,
        )
        results.append(success)

    # Summary for multiple files
    if len(pdf_files) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        successful = sum(results)
        failed = len(results) - successful
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

    # Exit with error if any failed
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
