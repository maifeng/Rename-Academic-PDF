"""Rename Academic PDF - Metadata-based PDF renaming for academic papers.

This package automatically renames academic PDFs using metadata from scholarly databases.
It queries multiple APIs (CrossRef, DataCite, arXiv, Semantic Scholar, PubMed, OpenAlex, DBLP)
to extract accurate paper metadata and generate standardized filenames.

Author: Feng Mai <maifeng@gmail.com>
"""

__version__ = "1.0.0"
__author__ = "Feng Mai"
__email__ = "maifeng@gmail.com"

from .core import (
    rename_pdf,
    extract_metadata_cascade,
    format_filename_from_metadata,
    abbreviate_journal,
)

__all__ = [
    "rename_pdf",
    "extract_metadata_cascade",
    "format_filename_from_metadata",
    "abbreviate_journal",
    "__version__",
]
