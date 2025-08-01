"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import json
from typing import Any, Optional, cast

import aiohttp
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State, DocumentInfo, DocumentStructure
from enrichment_agent.utils import (
    validate_document,
    extract_title_with_vision,
    find_toc_pages,
    extract_toc_from_page_with_vision,
    extract_bibliography_full_pipeline,
    init_model
)


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


_INFO_PROMPT = """You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

<Website content>
{content}
</Website content>"""


async def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Scrape and summarize content from a given URL.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    p = _INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        url=url,
        content=content[:40_000],
    )
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    return str(result.content)


async def validate_document_tool(
    *,
    pdf_path: str,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Validate a PDF document and update state with basic metadata."""
    result = validate_document(pdf_path)
    if not state.document_info:
        state.document_info = DocumentInfo()
    state.document_info.path = pdf_path
    state.document_info.file_type = "PDF"
    state.document_info.title = result["metadata"].get("title")
    state.document_info.publication_date = result["metadata"].get("publication_date")
    return result

async def extract_title_tool(
    *,
    pdf_path: str,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Extract document title using vision model and update state."""
    title = extract_title_with_vision(pdf_path, config)
    if not state.document_info:
        state.document_info = DocumentInfo()
    state.document_info.title = title
    return {"title": title}

async def extract_toc_tool(
    *,
    pdf_path: str,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Extract table of contents and update state."""
    toc_pages = find_toc_pages(pdf_path)
    toc_entries = []
    for page in toc_pages:
        toc_entries.extend(extract_toc_from_page_with_vision(pdf_path, page, config))
    if not state.document_structure:
        state.document_structure = DocumentStructure()
    state.document_structure.table_of_contents = toc_entries
    return {"toc": toc_entries}

async def extract_references_tool(
    *,
    pdf_path: str,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Extract references/bibliography and update state."""
    references = extract_bibliography_full_pipeline(pdf_path, config)
    if not state.document_structure:
        state.document_structure = DocumentStructure()
    state.document_structure.references = references
    return {"references": references}


