"""Utility functions for document processing."""

import base64
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import fitz

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from enrichment_agent.configuration import Configuration
from enrichment_agent import prompts

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"🔑 Environment loaded. ANTHROPIC_API_KEY: {'ANTHROPIC_API_KEY' in os.environ}, OPENAI_API_KEY: {'OPENAI_API_KEY' in os.environ}")
except ImportError:
    print("⚠️ python-dotenv not available, skipping .env loading")



def validate_document(pdf_path: str) -> Dict[str, any]:
    """
    Validate a PDF document before parsing. Returns validation results and metadata.
    """
    validation_result = {
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "metadata": {}
    }
    try:
        if not os.path.exists(pdf_path):
            validation_result["errors"].append(f"File not found: {pdf_path}")
            return validation_result
        file_size = os.path.getsize(pdf_path)
        validation_result["metadata"]["file_size_mb"] = round(file_size / (1024 * 1024), 2)
        if file_size > 50 * 1024 * 1024:
            validation_result["warnings"].append(f"Large file size: {validation_result['metadata']['file_size_mb']}MB")
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        validation_result["metadata"]["page_count"] = page_count
        if page_count == 0:
            validation_result["errors"].append("PDF has no pages")
            doc.close()
            return validation_result
        first_page = doc[0]
        test_text = first_page.get_text()
        validation_result["metadata"]["first_page_char_count"] = len(test_text)
        if len(test_text.strip()) < 10:
            validation_result["warnings"].append("Very little text on first page - may be scanned/image-based")
        try:
            metadata = doc.metadata
            validation_result["metadata"]["title"] = metadata.get("title", "")
            validation_result["metadata"]["author"] = metadata.get("author", "")
        except:
            validation_result["warnings"].append("Could not access document metadata")
        doc.close()
        validation_result["is_valid"] = True
        print(f"✅ Document validation passed: {page_count} pages, {validation_result['metadata']['first_page_char_count']} chars on page 1")
        if validation_result["warnings"]:
            print(f"⚠️ Warnings: {', '.join(validation_result['warnings'])}")
    except Exception as e:
        validation_result["errors"].append(f"PDF validation failed: {str(e)}")
        print(f"Document validation failed: {str(e)}")
    return validation_result

def get_message_text(msg: AnyMessage) -> str:
    """Return the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()

def init_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize the main language model for the agent."""
    configuration = Configuration.from_runnable_config(config)
    fully_specified_name = configuration.model
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)

def init_vision_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize a vision-capable model for visual document analysis."""
    configuration = Configuration.from_runnable_config(config)
    model_name = configuration.vision_model
    api_key_present = False
    if "anthropic" in model_name.lower():
        api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
        print(f"🔍 Anthropic API key present: {api_key_present}")
        if not api_key_present:
            print("💡 Tip: Set ANTHROPIC_API_KEY in your .env file")
    elif "openai" in model_name.lower():
        api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
        print(f"🔍 OpenAI API key present: {api_key_present}")
        if not api_key_present:
            print("💡 Tip: Set OPENAI_API_KEY in your .env file")
    if "/" in model_name:
        provider, model = model_name.split("/", maxsplit=1)
    else:
        provider = None
        model = model_name
    print(f"🤖 Initializing vision model: {model_name}")
    return init_chat_model(model, model_provider=provider)

def extract_first_page_as_image(pdf_path: str, page_num: int = 0) -> bytes:
    """
    Convert the first page of a PDF to PNG image data.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            raise ValueError("PDF has no pages")
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        image_data = pix.tobytes("png")
        pix = None
        doc.close()
        return image_data
    except Exception as e:
        raise ValueError(f"Failed to extract first page as image from {pdf_path}: {str(e)}")

def extract_title_with_vision(pdf_path: str, config: Optional[RunnableConfig] = None) -> str:
    """
    Extract document title using a vision model on the first page image.
    """
    try:
        image_data = extract_first_page_as_image(pdf_path, 0)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        vision_model = init_vision_model(config)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompts.VISION_TITLE_EXTRACTION_PROMPT
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        )
        response = vision_model.invoke([message])
        title = str(response.content).strip()
        if title and len(title) > 3:
            title = title.replace("Title:", "").strip()
            title = title.replace('"', "").strip()
            title = title.replace("'", "'").strip()
            if len(title) > 200:
                title = title[:200].strip()
            return title
        else:
            return "Untitled Document"
    except Exception as e:
        print(f"⚠️ Vision title extraction failed: {e}")
        return "Untitled Document"

def find_toc_pages(pdf_path: str, max_pages_to_scan: int = 15) -> List[int]:
    """
    Find pages containing a Table of Contents using text extraction.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    toc_pages = []
    toc_keywords = [
        'table of contents',
        'contents', 
        'toc'
    ]
    try:
        doc = fitz.open(pdf_path)
        pages_to_check = min(max_pages_to_scan, len(doc))
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text().lower()
            for keyword in toc_keywords:
                if keyword in text:
                    page_number_1_based = page_num + 1
                    if page_number_1_based not in toc_pages:
                        toc_pages.append(page_number_1_based)
                        print(f"📖 Found TOC keyword '{keyword}' on page {page_number_1_based}")
                    break
        doc.close()
        return sorted(toc_pages)
    except Exception as e:
        print(f"❌ Error finding TOC pages: {e}")
        return []

def extract_page_as_image(pdf_path: str, page_num: int) -> bytes:
    """
    Convert any page of a PDF to PNG image data.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            raise ValueError(f"Invalid page number {page_num}. Document has {len(doc)} pages.")
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        image_data = pix.tobytes("png")
        pix = None
        doc.close()
        return image_data
    except Exception as e:
        raise ValueError(f"Failed to extract page {page_num} as image from {pdf_path}: {str(e)}")

def extract_toc_from_page_with_vision(pdf_path: str, page_num: int, config: Optional[RunnableConfig] = None) -> List[Dict]:
    """
    Extract TOC entries from a page using a vision model.
    """
    try:
        print(f"🔍 Extracting TOC from page {page_num}...")
        image_data = extract_page_as_image(pdf_path, page_num)
        print(f"📸 Page {page_num} extracted as image: {len(image_data)} bytes")
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        vision_model = init_vision_model(config)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompts.VISION_TOC_EXTRACTION_PROMPT
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        )
        print(f"🤖 Analyzing page {page_num} with vision model...")
        response = vision_model.invoke([message])
        response_text = str(response.content).strip()
        print(f"📝 Raw response: {response_text[:200]}...")
        try:
            toc_entries = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text}")
            return []
        if not isinstance(toc_entries, list):
            print(f"⚠️ Expected list, got {type(toc_entries)}")
            return []
        valid_entries = []
        for entry in toc_entries:
            if isinstance(entry, dict) and all(key in entry for key in ['title', 'page', 'level']):
                entry['source_page'] = page_num
                valid_entries.append(entry)
            else:
                print(f"⚠️ Invalid TOC entry: {entry}")
        print(f"✅ Successfully extracted {len(valid_entries)} TOC entries from page {page_num}")
        return valid_entries
    except Exception as e:
        print(f"⚠️ Vision TOC extraction failed for page {page_num}: {e}")
        return []

def find_bibliography_page_from_toc(toc_entries: List[Dict]) -> Optional[int]:
    """
    Find the bibliography/references page number from TOC entries.
    """
    bibliography_keywords = [
        'references', 'bibliography', 'works cited', 
        'literature cited', 'sources', 'citations'
    ]
    for entry in toc_entries:
        title = entry.get('title', '').lower()
        for keyword in bibliography_keywords:
            if keyword in title:
                page_num = entry.get('page')
                if page_num and page_num > 0:
                    print(f"📚 Found bibliography '{entry['title']}' on page {page_num}")
                    return page_num
    print("⚠️ No bibliography section found in TOC")
    return None

def extract_text_pymupdf_page53(pdf_path: str) -> str:
    """
    Extract text from page 53 using PyMuPDF.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return ""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) < 53:
            print(f"❌ Document only has {len(doc)} pages, page 53 not available")
            doc.close()
            return ""
        print(f"📖 PyMuPDF: Extracting page 53")
        page = doc[51]
        text = page.get_text()
        doc.close()
        print(f"✅ PyMuPDF extracted {len(text)} characters from page 53")
        return text
    except Exception as e:
        print(f"❌ PyMuPDF extraction failed: {e}")
        return ""

def compare_page53_extraction(pdf_path: str = "inputs/MA_Nepal_2020.pdf") -> Dict[str, str]:
    """
    Extract text using PyMuPDF from page 53 (bibliography).
    """
    print("🔄 Extracting text using PyMuPDF from page 53 (bibliography)")
    print("=" * 80)
    results = {}
    print("\n🔧 Testing PyMuPDF...")
    results['pymupdf'] = extract_text_pymupdf_page53(pdf_path)
    print("\n" + "=" * 80)
    print("📊 EXTRACTION SUMMARY:")
    print("=" * 80)
    for method, text in results.items():
        char_count = len(text)
        line_count = len(text.split('\n')) if text else 0
        print(f"{method:12} | {char_count:6} chars | {line_count:4} lines")
    print("=" * 80)
    return results

def extract_bibliography_text_from_toc(pdf_path: str, toc_entries: List[Dict], max_pages: int = 3) -> str:
    """
    Extract bibliography text using PyMuPDF based on TOC-identified pages.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return ""
    bib_page = find_bibliography_page_from_toc(toc_entries)
    if not bib_page:
        print("⚠️ No bibliography section found in TOC, defaulting to page 53")
        bib_page = 53
    print(f"📚 Extracting bibliography starting from page {bib_page}")
    try:
        doc = fitz.open(pdf_path)
        if bib_page > len(doc):
            print(f"❌ Bibliography page {bib_page} exceeds document length ({len(doc)} pages)")
            doc.close()
            return ""
        all_text = []
        end_page = min(bib_page + max_pages - 1, len(doc))
        print(f"📖 PyMuPDF: Extracting bibliography pages {bib_page}-{end_page}")
        for page_num in range(bib_page - 1, end_page):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                all_text.append(f"=== PAGE {page_num + 1} ===\n{text}\n")
        doc.close()
        combined_text = "\n".join(all_text)
        print(f"✅ PyMuPDF extracted {len(combined_text)} characters from {end_page - bib_page + 1} pages")
        return combined_text
    except Exception as e:
        print(f"❌ Bibliography text extraction failed: {e}")
        return ""

def parse_bibliography_with_llm(raw_text: str, config: Optional[RunnableConfig] = None) -> List[Dict]:
    """
    Parse raw bibliography text into structured entries using an LLM.
    """
    if not raw_text.strip():
        print("❌ No text provided for bibliography parsing")
        return []
    try:
        print(f"🤖 Parsing bibliography with GPT-4.1-mini ({len(raw_text)} chars)...")
        model = init_chat_model("gpt-4.1-mini-2025-04-14", model_provider="openai")
        bibliography_prompt = prompts.BIBLIOGRAPHY_PARSING_PROMPT.format(bibliography_text=raw_text)
        message = HumanMessage(content=bibliography_prompt)
        response = model.invoke([message])
        response_text = str(response.content).strip()
        print(f"📝 LLM response: {response_text[:200]}...")
        try:
            bibliography_entries = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON response: {e}")
            print(f"📄 Full response length: {len(response_text)} chars")
            print(f"📄 Response preview: {response_text[:500]}...")
            return []
        if not isinstance(bibliography_entries, list):
            print(f"⚠️ Expected list, got {type(bibliography_entries)}")
            return []
        print(f"✅ Successfully parsed {len(bibliography_entries)} bibliography entries")
        return bibliography_entries
    except Exception as e:
        print(f"❌ LLM bibliography parsing failed: {e}")
        return []

def extract_bibliography_full_pipeline(pdf_path: str, config: Optional[RunnableConfig] = None) -> List[Dict]:
    """
    Complete bibliography extraction pipeline: TOC → PyMuPDF → LLM parsing.
    """
    print("🔄 Starting complete bibliography extraction pipeline")
    print("=" * 80)
    try:
        print("📖 Step 1: Finding TOC pages...")
        toc_pages = find_toc_pages(pdf_path)
        if not toc_pages:
            print("⚠️ No TOC pages found, will use default bibliography location")
            toc_entries = []
        else:
            print(f"✅ Found TOC on pages: {toc_pages}")
            print("🤖 Step 2: Extracting TOC entries with vision...")
            toc_entries = []
            for toc_page in toc_pages:
                page_entries = extract_toc_from_page_with_vision(pdf_path, toc_page, config)
                toc_entries.extend(page_entries)
            print(f"✅ Extracted {len(toc_entries)} total TOC entries")
        print("📚 Step 3: Extracting bibliography text with PyMuPDF...")
        raw_bib_text = extract_bibliography_text_from_toc(pdf_path, toc_entries)
        if not raw_bib_text:
            print("❌ No bibliography text extracted")
            return []
        print("🤖 Step 4: Parsing bibliography with LLM...")
        bibliography_entries = parse_bibliography_with_llm(raw_bib_text, config)
        print("\n" + "=" * 80)
        print("📊 BIBLIOGRAPHY EXTRACTION SUMMARY:")
        print("=" * 80)
        print(f"TOC pages found: {len(toc_pages)}")
        print(f"TOC entries extracted: {len(toc_entries)}")
        print(f"Bibliography text length: {len(raw_bib_text)} characters")
        print(f"Bibliography entries parsed: {len(bibliography_entries)}")
        print("=" * 80)
        if bibliography_entries:
            print(f"\n✅ Bibliography extraction completed successfully!")
            print(f"📚 Found {len(bibliography_entries)} bibliography entries")
            print(f"\n📝 Preview of first 3 entries:")
            for i, entry in enumerate(bibliography_entries[:3], 1):
                name = entry.get('name', 'Unknown Name')[:60]
                year = entry.get('year', 'Unknown Year')
                link = entry.get('link', 'No link')
                print(f"   {i}. {name}... ({year}) - {link}")
        else:
            print("❌ No bibliography entries were successfully parsed")
        return bibliography_entries
    except Exception as e:
        print(f"❌ Bibliography extraction pipeline failed: {e}")
        return []

def extract_tables_from_page(pdf_path: str, page_num: int) -> List[Dict]:
    """
    Extract tables from a specific page using pdfplumber.
    """
    try:
        import pdfplumber
    except ImportError:
        print(f"❌ pdfplumber not installed. Install with: pip install pdfplumber")
        return []
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return []
    try:
        print(f"📊 Extracting tables from page {page_num} using pdfplumber")
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                print(f"❌ Invalid page number {page_num}. Document has {len(pdf.pages)} pages")
                return []
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables()
            if not tables:
                print(f"⚠️ No tables found on page {page_num}")
                return []
            print(f"✅ Found {len(tables)} table(s) on page {page_num}")
            extracted_tables = []
            for i, table_data in enumerate(tables, 1):
                print(f"📋 Processing table {i}...")
                if not table_data:
                    print(f"   ⚠️ Table {i} is empty")
                    continue
                rows = len(table_data)
                columns = len(table_data[0]) if table_data else 0
                page_width = page.width
                page_height = page.height
                table_info = {
                    "table_number": i,
                    "page": page_num,
                    "library": "pdfplumber",
                    "bbox": {
                        "x0": 0,
                        "y0": 0,
                        "x1": page_width,
                        "y1": page_height
                    },
                    "rows": rows,
                    "columns": columns,
                    "data": table_data
                }
                extracted_tables.append(table_info)
                print(f"   📏 Table {i}: {rows} rows x {columns} columns")
            return extracted_tables
    except Exception as e:
        print(f"❌ pdfplumber table extraction failed: {e}")
        return []


def extract_all_tables_from_pdf(pdf_path: str, max_pages: int = None) -> Dict[int, List[Dict]]:
    """
    Extract all tables from a PDF document using pdfplumber.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return {}
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_scan = min(max_pages, total_pages) if max_pages else total_pages
            print(f"🔄 Scanning {pages_to_scan} pages for tables using pdfplumber...")
            all_tables = {}
            total_table_count = 0
            for page_num in range(1, pages_to_scan + 1):
                tables = extract_tables_from_page(pdf_path, page_num)
                if tables:
                    all_tables[page_num] = tables
                    total_table_count += len(tables)
                    print(f"   📊 Page {page_num}: {len(tables)} table(s)")
            print(f"\n✅ Scan complete: {total_table_count} tables found across {len(all_tables)} pages")
            return all_tables
    except Exception as e:
        print(f"❌ PDF table scan failed: {e}")
        return {}


def detect_tables_by_text_analysis(pdf_path: str, page_num: int, min_columns: int = 3) -> List[Dict]:
    """
    Detect table-like structures by analyzing text patterns as a fallback method.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return []
    try:
        import fitz
        print(f"🔍 Analyzing text patterns on page {page_num}")
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        words = page.get_text("words")
        if not words:
            print(f"⚠️ No text found on page {page_num}")
            doc.close()
            return []
        print(f"📝 Analyzing {len(words)} words for table patterns...")
        rows = {}
        for word in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word
            y_rounded = round(y0, 1)
            if y_rounded not in rows:
                rows[y_rounded] = []
            rows[y_rounded].append((x0, text.strip()))
        potential_table_rows = []
        for y in sorted(rows.keys()):
            row_words = rows[y]
            if len(row_words) >= min_columns:
                row_words.sort()
                potential_table_rows.append({
                    "y_position": y,
                    "words": row_words,
                    "columns": len(row_words)
                })
        if not potential_table_rows:
            print(f"⚠️ No table patterns found (need at least {min_columns} columns)")
            doc.close()
            return []
        tables = []
        current_table_rows = []
        prev_y = None
        y_threshold = 20
        for row in potential_table_rows:
            y = row["y_position"]
            if prev_y is None or abs(y - prev_y) <= y_threshold:
                current_table_rows.append(row)
            else:
                if len(current_table_rows) >= 2:
                    tables.append(current_table_rows)
                current_table_rows = [row]
            prev_y = y
        if len(current_table_rows) >= 2:
            tables.append(current_table_rows)
        print(f"✅ Found {len(tables)} potential table(s)")
        extracted_tables = []
        for i, table_rows in enumerate(tables, 1):
            table_data = []
            max_columns = max(row["columns"] for row in table_rows)
            for row in table_rows:
                row_data = [word[1] for word in row["words"]]
                while len(row_data) < max_columns:
                    row_data.append(None)
                table_data.append(row_data)
            all_words = []
            for row in table_rows:
                all_words.extend(row["words"])
            min_x = min(word[0] for word in all_words)
            max_x = max(word[0] for word in all_words)
            min_y = min(row["y_position"] for row in table_rows)
            max_y = max(row["y_position"] for row in table_rows)
            table_info = {
                "table_number": i,
                "page": page_num,
                "library": "text_analysis",
                "bbox": {
                    "x0": min_x,
                    "y0": min_y,
                    "x1": max_x,
                    "y1": max_y
                },
                "rows": len(table_data),
                "columns": max_columns,
                "data": table_data
            }
            extracted_tables.append(table_info)
            print(f"   📋 Table {i}: {len(table_data)} rows x {max_columns} columns")
        doc.close()
        return extracted_tables
    except Exception as e:
        print(f"❌ Text-based table detection failed: {e}")
        return []
    
    
def extract_metadata_from_user_query(user_query: str) -> Dict:
    """
    Extract metadata from the user query.
    """
    metadata = {}
    
    try:
        print(f"Extracting metadata from user query: {user_query}")
        model = init_chat_model("gpt-4.1-nano-2025-04-14", model_provider="openai")
        metadata_prompt = prompts.METADATA_EXTRACTION_PROMPT.format(user_query=user_query, current_year=datetime.now().year)
        message = HumanMessage(content=metadata_prompt)
        response = model.invoke([message])
        metadata = json.loads(response.content)
    except Exception as e:
        print(f"❌ Metadata extraction failed: {e}")
        return {}
    
    return metadata


def generate_search_query(user_query: str, reference: dict) -> str:
    '''
    Generate a search query based on the metadata and reference.
    ''' 
    
    name, year, link = reference.get("name"), reference.get("year"), reference.get("link")
    metadata = extract_metadata_from_user_query(user_query)
    query_year = metadata.get("year")
    
    cleaned_name = re.sub(r"\b(20\d{2}|19\d{2})\b", "", name).strip()
    
    query_parts = [cleaned_name]
    if query_year:
        query_parts.append(str(query_year))
    query_parts.extend([str(v) for k, v in metadata.items() if v and k != "year" ])
    search_query = " ".join(query_parts).strip()
    return search_query
