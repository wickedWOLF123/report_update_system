"""Define a document analysis agent.

Works with document processing tools to extract and analyze PDF content.
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import InputState, OutputState, State
from enrichment_agent.tools import (
    validate_document_tool, extract_title_tool, extract_toc_tool, extract_references_tool,
    extract_tables_tool, extract_all_tables_tool, detect_tables_text_analysis_tool,
    extract_metadata_from_query_tool, generate_search_query_tool
)


async def document_analysis(
    state: State, *, config: Optional[RunnableConfig] = None
) -> State:
    """Run document validation, title, TOC, and references extraction, updating the state."""
    pdf_path = state.document_path
    if not pdf_path:
        print("❌ No document_path provided in state!")
        return state
    
    print(f"📄 Starting document analysis for: {pdf_path}")
    print(f"🎯 User topic: {state.topic}")
    
    # Step 1: Extract metadata from user query
    try:
        await extract_metadata_from_query_tool(user_query=state.topic, state=state, config=config)
        print(f"✅ Extracted metadata: {state.document_info.metadata if state.document_info else 'None'}")
    except Exception as e:
        print(f"⚠️ extract_metadata_from_query_tool failed: {e}")
    
    # Step 2: Validate document
    try:
        await validate_document_tool(pdf_path=pdf_path, state=state, config=config)
        print(f"✅ Document validation complete")
    except Exception as e:
        print(f"⚠️ validate_document_tool failed: {e}")
    
    # Step 3: Extract title
    try:
        await extract_title_tool(pdf_path=pdf_path, state=state, config=config)
        print(f"✅ Title extracted: {state.document_info.title if state.document_info else 'None'}")
    except Exception as e:
        print(f"⚠️ extract_title_tool failed: {e}")
    
    # Step 4: Extract TOC
    try:
        await extract_toc_tool(pdf_path=pdf_path, state=state, config=config)
        toc_count = len(state.document_structure.table_of_contents) if state.document_structure and state.document_structure.table_of_contents else 0
        print(f"✅ TOC extracted: {toc_count} entries")
    except Exception as e:
        print(f"⚠️ extract_toc_tool failed: {e}")
    
    # Step 5: Extract references
    try:
        await extract_references_tool(pdf_path=pdf_path, state=state, config=config)
        ref_count = len(state.document_structure.references) if state.document_structure and state.document_structure.references else 0
        print(f"✅ References extracted: {ref_count} entries")
    except Exception as e:
        print(f"⚠️ extract_references_tool failed: {e}")
    
    
    state.processing_stage = "document_analysis_complete"
    print("🎉 Document analysis completed successfully!")
    return state


async def finalize_results(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Prepare final results for output."""
    
    # Create a comprehensive summary of extracted information
    results = {
        "document_info": {
            "path": state.document_info.path if state.document_info else None,
            "title": state.document_info.title if state.document_info else None,
            "file_type": state.document_info.file_type if state.document_info else None,
            "page_count": state.document_info.page_count if state.document_info else None,
            "publication_date": state.document_info.publication_date if state.document_info else None,
            "metadata": state.document_info.metadata if state.document_info else None,
        },
        "document_structure": {
            "table_of_contents": state.document_structure.table_of_contents if state.document_structure else None,
            "references": state.document_structure.references if state.document_structure else None,
            "tables": state.document_structure.tables if state.document_structure else None,
        },
        "processing_stage": state.processing_stage,
        "user_topic": state.topic,
    }
    
    return {"info": results}


# Create the simplified graph
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)

# Add nodes
workflow.add_node("document_analysis", document_analysis)
workflow.add_node("finalize_results", finalize_results)

# Add edges
workflow.add_edge("__start__", "document_analysis")
workflow.add_edge("document_analysis", "finalize_results")
workflow.add_edge("finalize_results", "__end__")

# Compile the graph
graph = workflow.compile()
graph.name = "DocumentAnalysis"
