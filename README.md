# Rename Academic PDF Files

Tired of PDFs of academic papers with names like `hhaf081.pdf` or `1-s2.0-S0377221718308774-main.pdf`? This tool automatically renames academic PDFs to meaningful filenames using metadata from scholarly databases. For example:

```
rename-academic-pdf hhaf081.pdf

hhaf081.pdf → LiMaiShenYangZhang2025-Dissecting Corporate Culture Using Generative AI-RFS.pdf
```

The tool extracts DOIs, arXiv IDs, or paper titles from the PDF, queries multiple academic APIs (CrossRef, OpenAlex, Semantic Scholar, arXiv, PubMed, and more), and generates clean, consistent filenames with author names, publication year, title, and journal abbreviation.

## Quick Start

```bash
# Install
pip install rename-academic-pdf

# Rename a single PDF
rename-academic-pdf paper.pdf

# Batch rename
rename-academic-pdf *.pdf
rename-academic-pdf **/*.pdf  # Nested directories

# Preview without renaming
rename-academic-pdf paper.pdf --dry-run
```

## Installation

### From PyPI (recommended)

```bash
pip install rename-academic-pdf
```

### From source

```bash
git clone https://github.com/maifeng/rename-academic-pdf.git
cd rename-academic-pdf
pip install -e .
```

**Requirements**: Python 3.7+

---

## Features

- **Intelligent identifier extraction**: DOI, arXiv ID, PMID from file name, PDF text and metadata
- **Multi-API cascade**: Queries 7+ academic databases with smart fallbacks
- **Journal abbreviations**: Built-in abbreviations for 70+ top journals (FT50, UTD24)
- **Batch processing**: Rename multiple PDFs with wildcards (`*.pdf`, `**/*.pdf`)
- **No API key required**: Most APIs are free (optional Semantic Scholar key for better rate limits)

## Filename Formats

### Default Behavior
Default format: `AuthorsYear-Title-Journal.pdf`
- **≤ 5 authors**: All authors concatenated (e.g., `DoeDeo2023`)
- **> 5 authors**: First author + "EtAl" (e.g., `DoeEtAl2023`)

You can override the default format string using command line options or in a config file (see [Configuration File](#configuration-file) section).

    
### Format Presets

| Preset          | Template                            | Example                               |
| --------------- | ----------------------------------- | ------------------------------------- |
| `default`       | `{author}{year}-{title}-{journal}`  | `Author2025-PaperTitle-JournalName.pdf`  |
| `compact`       | `{author}{year}-{title}`            | `Author2025-PaperTitle.pdf`         |
| `full`          | `{author}-{year}-{title}-{journal}` | `Author-2025-PaperTitle-JournalName.pdf` |
| `minimal`       | `{author}{year}`                    | `Author2025.pdf`                       |
| `year_first`    | `{year}-{author}-{title}`           | `2025-Author-PaperTitle.pdf`        |
| `journal_first` | `{journal}-{author}{year}-{title}`  | `JournalName-Author2025-PaperTitle.pdf`  |

```bash
rename-academic-pdf paper.pdf --format compact      # No journal
rename-academic-pdf paper.pdf --format minimal      # Author + year only
rename-academic-pdf paper.pdf --format year_first   # Year first
```

### Custom Format Strings

Create your own format using template variables:
- `{author}` - Author name(s): all authors if ≤5, FirstAuthorEtAl if >5
- `{year}` - Publication year
- `{title}` - Paper title
- `{journal}` - Journal abbreviation

```bash
rename-academic-pdf paper.pdf --format-string '{journal}_{year}_{author}'
rename-academic-pdf paper.pdf --format-string '{author}-{title}'
```


### Additional Options

**`--first-author-only`**: Use only first author (reverts to old behavior)
```bash
rename-academic-pdf paper.pdf --first-author-only
# Output: AuthorA2023-Title-Journal.pdf (instead of AuthorAAuthorBAuthorC2023...)
```

**`--separator` (`-` or `_`)**: Change separator character
```bash
rename-academic-pdf paper.pdf --separator _
# Output: LiMaiShen2023_Title_Journal.pdf
```

**`--journal-abbrev-file`**: Use custom journal abbreviations file
```bash
rename-academic-pdf paper.pdf --journal-abbrev-file ~/my-journals.json
# Uses custom abbreviations from the specified JSON file
# Can be saved in ~/.rename-academic-pdf/journal_abbreviations.json for automatic loading
# See format: https://github.com/maifeng/rename-academic-pdf/blob/main/journal_abbreviations.json
```

**`--max-title-length`**: Maximum title length in filename (default: 80)
```bash
rename-academic-pdf paper.pdf --max-title-length 120
# Longer titles allowed (truncates at word boundary, never mid-word)
```

**`--bib-file`**: Append BibTeX entries to a file
```bash
rename-academic-pdf paper.pdf --bib-file ~/papers.bib
# Fetches BibTeX from DOI.org or arXiv, or generates from metadata
# Each entry includes comments with PDF and markdown paths
```

The BibTeX entries are fetched directly from authoritative sources when possible:
- **DOI**: Uses DOI.org content negotiation (`application/x-bibtex`)
- **arXiv**: Uses arXiv's BibTeX export endpoint
- **Fallback**: Generates BibTeX from extracted metadata using pybtex

**`--markdown-dir`**: Generate markdown versions of PDFs
```bash
rename-academic-pdf paper.pdf --markdown-dir ~/paper_markdown/
# Converts PDFs to markdown using markitdown
# Markdown files are saved with same name as renamed PDF
# Requires: pip install rename-academic-pdf[markdown] or pip install rename-academic-pdf[all]
```

## API Coverage

The script tries multiple APIs in cascade order:

### Identifier-Based (Primary)
1. **DOI** → DOI.org → CrossRef → DataCite → Semantic Scholar
2. **arXiv ID** → arXiv API → Semantic Scholar
3. **SSRN ID** → Convert to DOI (`10.2139/ssrn.{id}`) → DOI.org → CrossRef
4. **PMID** → PubMed API

### Title-Based (Fallback)
5. **Semantic Scholar** (200M+ papers, CS/AI focus)
6. **DBLP** (Computer science bibliography)
7. **OpenAlex** (200M+ papers, all fields)

### Database Coverage
- **DOI.org**: Authoritative DOI resolver (Citeproc JSON)
- **CrossRef**: 130M+ journal articles (including SSRN)
- **DataCite**: Datasets, conferences, grey literature
- **arXiv**: STEM preprints
- **SSRN**: Working papers (via DOI lookup)
- **PubMed**: Biomedical literature
- **Semantic Scholar**: CS/AI papers (optional API key)
- **DBLP**: Computer science papers
- **OpenAlex**: Comprehensive, free, no API key

## Environment Variables (Optional)

```bash
# ~/.bashrc or ~/.zshrc
export SEMANTIC_SCHOLAR_API_KEY="your-api-key-here"
export PUBMED_API_KEY="your-api-key-here"  # For faster rate limits
export EMAIL="your@email.com"  # For CrossRef polite pool
export OPENAI_API_KEY="your-api-key-here"  # For --llm flag (OpenAI)
export OPENROUTER_API_KEY="your-api-key-here"  # For --llm flag (OpenRouter)
```

Get a free Semantic Scholar API key: https://www.semanticscholar.org/product/api

## LLM-Based Extraction (Experimental)

When the `--llm` flag is enabled, the script will use an LLM as a **fallback** after all API-based methods fail. It extracts metadata from the first 3 pages of PDF text. This could be useful for working papers without doi. The default model is gpt-4.1-mini. Supports other OpenAI and OpenRouter model.

### OpenAI (Default)

```bash
# Uses OPENAI_API_KEY
rename-academic-pdf *.pdf --llm
rename-academic-pdf *.pdf --llm --llm-model gpt-4o-mini
```

### OpenRouter

Use `provider/model` format to automatically use OpenRouter:

```bash
# Uses OPENROUTER_API_KEY (auto-detected from model format)
rename-academic-pdf *.pdf --llm --llm-model anthropic/claude-3-haiku
rename-academic-pdf *.pdf --llm --llm-model google/gemini-2.0-flash-001
```

**Requirements:**
- `pip install openai` (or `pip install -e '.[llm]'`)
- `OPENAI_API_KEY` for OpenAI models, or `OPENROUTER_API_KEY` for OpenRouter models set in environment variables.


## Journal Abbreviations

The package includes built-in abbreviations for 100+ major academic journals. For example:
- "Journal of Management Information Systems" → "JMIS"
- "Information Systems Research" → "ISR"
- "Review of Financial Studies" → "RFS"

### Custom Journal Abbreviations

You can provide your own journal abbreviations to override or extend the built-in list. The package searches for custom abbreviation files in the following order:

1. **Command-line argument**: `--journal-abbrev-file path/to/file.json`
2. **User's home directory**: `~/.rename-academic-pdf/journal_abbreviations.json`
3. **Default bundled file**: Built-in abbreviations

#### Creating a Custom Abbreviations File

Create a JSON file with the following structure:

```json
{
    "comment": "My custom journal abbreviations",
    "abbreviations": {
        "Journal of Interesting Research": "JIR",
        "Quarterly Review of Examples": "QRE",
        "Proceedings of Example Conference": "PEC"
    }
}
```

#### Using Custom Abbreviations

**Option 1: Command-line argument**
```bash
rename-academic-pdf paper.pdf --journal-abbrev-file ~/my-journals.json
```

**Option 2: User home directory** (automatically loaded)
```bash
# Create the directory
mkdir -p ~/.rename-academic-pdf

# Copy or create your custom file
cp my-journals.json ~/.rename-academic-pdf/journal_abbreviations.json

# Run normally - custom abbreviations will be used automatically
rename-academic-pdf paper.pdf
```

## Configuration File

You can set default options by creating a config file at `~/.rename-academic-pdf/config.json`:

```json
{
    "format_string": "{author}_{year}_{journal}_{title}",
    "first_author_only": true,
    "max_title_length": 100,
    "llm": true,
    "llm_model": "gpt-4o-mini",
    "bib_file": "~/papers.bib",
    "markdown_dir": "~/paper_markdown"
}
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `format` | string | `"default"` | Format preset (`default`, `compact`, `full`, `minimal`, `year_first`, `journal_first`) |
| `format_string` | string | - | Custom format string (overrides `format` if both set) |
| `separator` | string | `"-"` | Separator character (`"-"` or `"_"`) |
| `first_author_only` | boolean | `false` | Use only first author |
| `max_title_length` | integer | `80` | Maximum title length in filename (truncates at word boundary) |
| `llm` | boolean | `false` | Enable LLM fallback |
| `llm_model` | string | `"gpt-4.1-mini"` | LLM model for `--llm` mode |
| `bib_file` | string | - | Path to BibTeX file to append entries to |
| `markdown_dir` | string | - | Directory to save markdown versions of PDFs |

Command-line arguments always override config file settings.

## License

MIT License - see LICENSE file

## Author

Created by [Feng Mai](https://fengmai.net). 

☕ If this tool saved you time, consider [buying me a coffee](https://buymeacoffee.com/fengmai) 