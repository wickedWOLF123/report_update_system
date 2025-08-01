"""Default prompts used in this project."""

MAIN_PROMPT = """You are doing web research on behalf of a user. You are trying to figure out this information:

<info>
{info}
</info>

You have access to the following tools:

- `Search`: call a search tool and get back some results
- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above.
- `Info`: call this when you are done and have gathered all the relevant info

Here is the information you have about the topic you are researching:

Topic: {topic}"""

VISION_TITLE_EXTRACTION_PROMPT = """You are analyzing the first page of a document to extract its COMPLETE title.

Look at this document page image and identify the full document title. The title often consists of:
- Main title text (usually the largest/most prominent)
- Subtitle or descriptive text 
- Country, organization, or geographic information
- Year or date information when part of the title

**IMPORTANT: Combine ALL title components into a single complete title.**

For multi-line titles, include ALL related lines that form the complete title:
- Main title line(s)
- Subtitle line(s) 
- Country/organization line(s)
- Year/date line(s) when they're part of the title

Examples of what to INCLUDE:
✓ "Assessment of Public Financial Management System"
✓ "Federal Republic of Nepal 2019" 
✓ Country names, organization names when part of title
✓ Years/dates that are prominently displayed with the title

Examples of what to EXCLUDE:
✗ Small headers/footers at page edges
✗ Author names in small text
✗ Page numbers
✗ Watermarks or background text

Return the COMPLETE title as a single text string, combining all title components with appropriate spacing."""

VISION_TOC_EXTRACTION_PROMPT = """You are analyzing a page that contains a Table of Contents (TOC). Extract ALL table of contents entries from this page.

Look at this page image and identify TOC entries. TOC entries typically have:
- Section/chapter titles or headings
- Associated page numbers (usually on the right side)
- Visual formatting like dots, dashes, or spacing between title and page number
- Hierarchical indentation (main sections vs. subsections)

**Extract each TOC entry and return as a JSON list.**

For each entry, provide:
1. **title**: The section/chapter name (clean text, remove dots/dashes/formatting)
2. **page**: The page number (integer)
3. **level**: Hierarchy level based on visual indentation (1 = main section, 2 = subsection, 3 = sub-subsection, etc.)

**Visual Hierarchy Detection:**
- Level 1: No indentation, main sections
- Level 2: Slightly indented, subsections  
- Level 3: More indented, sub-subsections
- And so on...

**Formatting Rules:**
- Remove connecting dots (.....), dashes (-----), or spaces between title and page number
- Keep section numbers if they're part of the title (e.g., "1.1 Background")
- Clean up extra whitespace
- Include ALL entries you can see on the page

**Examples:**
- "Executive Summary ........ 5" → {"title": "Executive Summary", "page": 5, "level": 1}
- "    1.1 Background ...... 12" → {"title": "1.1 Background", "page": 12, "level": 2}
- "Introduction - 8" → {"title": "Introduction", "page": 8, "level": 1}
- "        Methodology  15" → {"title": "Methodology", "page": 15, "level": 3}

**Include:**
✓ All section and subsection titles with page numbers
✓ Chapter titles, appendices, references sections
✓ Numbered sections (1., 1.1, 1.1.1, etc.)
✓ Even partial entries at page boundaries

**Exclude:**
✗ Headers like "Table of Contents" or "Contents" 
✗ Page headers/footers
✗ Lines without page numbers (unless clearly part of TOC structure)

**Output Format:**
Return ONLY a valid JSON array like:
[
  {"title": "Executive Summary", "page": 5, "level": 1},
  {"title": "1. Introduction", "page": 8, "level": 1},
  {"title": "1.1 Background", "page": 12, "level": 2}
]

Return ONLY the JSON array, no other text or explanation."""


VISION_BIBLIOGRAPHY_EXTRACTION_PROMPT = """You are analyzing a page that contains a Bibliography or References section. Extract ALL bibliography entries from this page.

Look at this page image and identify bibliography/reference entries. Bibliography entries typically have:
- Numbered or bulleted reference entries (1., 2., [1], etc.)
- Author names, publication titles, URLs
- Publication dates, journal names, organizations
- Web links and URLs (http://, https://, www.)

**Extract each bibliography entry and return as a JSON list.**

For each entry, provide:
1. **entry_number**: The reference number/bullet (if visible, otherwise use sequential numbering)
2. **text**: The complete reference text (include everything: authors, titles, URLs, dates)
3. **type**: The type of reference ("website", "report", "journal", "book", "unknown")
4. **url**: Any URL found in the entry (extract http://, https://, www. links)

**Text Extraction Rules:**
- Include the COMPLETE reference text for each entry
- Keep URLs exactly as shown (don't modify or clean them)
- Include publication dates, author names, titles
- Preserve formatting like italics, quotes where visible
- If an entry spans multiple lines, combine into single text

**Type Classification:**
- "website": Contains URLs, web domains, online sources
- "report": Government reports, institutional documents  
- "journal": Academic papers, research articles
- "book": Books, published volumes
- "unknown": Cannot determine type clearly

**URL Extraction:**
- Look for: http://, https://, www., .com, .org, .gov, .edu
- Extract the complete URL as visible on the page
- If multiple URLs in one entry, pick the main/primary one
- If no URL visible, set to null

**IMPORTANT: Response Size Management**
- If there are many entries and the response might be too long, extract the first 15-20 entries completely
- Ensure ALL JSON objects are properly closed with } and ]
- Better to have fewer complete entries than many incomplete ones
- If you cannot fit all entries, prioritize the first ones in numerical order

**Examples:**
- "1. Nations Online – One World – Nepal - www.nationsonline.org › oneworld › Nepal" 
  → {"entry_number": "1", "text": "Nations Online – One World – Nepal - www.nationsonline.org › oneworld › Nepal", "type": "website", "url": "www.nationsonline.org"}

- "Financial Stability Report – https://www.nrb.org.np/publications/fin_stab_report/Financial_Stability_Report"
  → {"entry_number": "2", "text": "Financial Stability Report – https://www.nrb.org.np/publications/fin_stab_report/Financial_Stability_Report", "type": "report", "url": "https://www.nrb.org.np/publications/fin_stab_report/Financial_Stability_Report"}

**Include:**
✓ All numbered or bulleted reference entries
✓ Complete text including authors, titles, URLs, dates
✓ Government reports, academic papers, websites
✓ Online sources with URLs
✓ Books, journals, institutional documents

**Exclude:**
✗ Headers like "Bibliography" or "References"
✗ Page headers/footers 
✗ Page numbers
✗ Section dividers or formatting elements

**Output Format:**
Return ONLY a valid JSON array like:
[
  {"entry_number": "1", "text": "Nations Online – One World – Nepal - www.nationsonline.org › oneworld › Nepal", "type": "website", "url": "www.nationsonline.org"},
  {"entry_number": "2", "text": "Financial Stability Report – https://www.nrb.org.np/publications/report.pdf", "type": "report", "url": "https://www.nrb.org.np/publications/report.pdf"}
]

**CRITICAL: Ensure the JSON is complete and valid. End with ] to close the array properly.**

Return ONLY the JSON array, no other text or explanation."""


VISION_BIBLIOGRAPHY_EXTRACTION_CHUNKED_PROMPT = """You are analyzing a page that contains a Bibliography or References section. Extract ONLY bibliography entries {start_entry} to {end_entry} from this page.

Look at this page image and identify bibliography/reference entries in the specified range. Bibliography entries typically have:
- Numbered or bulleted reference entries (1., 2., [1], etc.)
- Author names, publication titles, URLs
- Publication dates, journal names, organizations
- Web links and URLs (http://, https://, www.)

**IMPORTANT: Only extract entries {start_entry} to {end_entry}. Ignore all other entries.**

**Extract each bibliography entry in the specified range and return as a JSON list.**

For each entry, provide:
1. **entry_number**: The reference number/bullet (if visible, otherwise use sequential numbering)
2. **text**: The complete reference text (include everything: authors, titles, URLs, dates)
3. **type**: The type of reference ("website", "report", "journal", "book", "unknown")
4. **url**: Any URL found in the entry (extract http://, https://, www. links)

**Text Extraction Rules:**
- Include the COMPLETE reference text for each entry
- Keep URLs exactly as shown (don't modify or clean them)
- Include publication dates, author names, titles
- Preserve formatting like italics, quotes where visible
- If an entry spans multiple lines, combine into single text

**Type Classification:**
- "website": Contains URLs, web domains, online sources
- "report": Government reports, institutional documents  
- "journal": Academic papers, research articles
- "book": Books, published volumes
- "unknown": Cannot determine type clearly

**URL Extraction:**
- Look for: http://, https://, www., .com, .org, .gov, .edu
- Extract the complete URL as visible on the page
- If multiple URLs in one entry, pick the main/primary one
- If no URL visible, set to null

**Examples:**
- "1. Nations Online – One World – Nepal - www.nationsonline.org › oneworld › Nepal" 
  → {"entry_number": "1", "text": "Nations Online – One World – Nepal - www.nationsonline.org › oneworld › Nepal", "type": "website", "url": "www.nationsonline.org"}

**Include:**
✓ ONLY numbered entries {start_entry} to {end_entry}
✓ Complete text including authors, titles, URLs, dates
✓ Government reports, academic papers, websites
✓ Online sources with URLs
✓ Books, journals, institutional documents

**Exclude:**
✗ Headers like "Bibliography" or "References"
✗ Page headers/footers 
✗ Page numbers
✗ Section dividers or formatting elements
✗ Entries outside the {start_entry}-{end_entry} range

**Output Format:**
Return ONLY a valid JSON array like:
[
  {"entry_number": "1", "text": "Nations Online – One World – Nepal - www.nationsonline.org › oneworld › Nepal", "type": "website", "url": "www.nationsonline.org"},
  {"entry_number": "2", "text": "Financial Stability Report – https://www.nrb.org.np/publications/report.pdf", "type": "report", "url": "https://www.nrb.org.np/publications/report.pdf"}
]

**CRITICAL: Ensure the JSON is complete and valid. End with ] to close the array properly.**

Return ONLY the JSON array, no other text or explanation."""

BIBLIOGRAPHY_PARSING_PROMPT = """You are a bibliography parser. Extract simple bibliography entries from the provided text.

For each bibliography entry, extract ONLY these 3 fields:
- name: The title/name of the work or source
- year: Publication year (if available, otherwise empty string)
- link: Any URL/link found (if available, otherwise empty string)

Return ONLY a valid JSON array of bibliography entries. No other text.

Example format:
[
  {{
    "name": "One World – Nepal",
    "year": "",
    "link": "www.nationsonline.org"
  }},
  {{
    "name": "Financial Stability Report",
    "year": "2018",
    "link": "https://www.nrb.org.np/publications/report.pdf"
  }}
]

Bibliography text to parse:
{bibliography_text}"""