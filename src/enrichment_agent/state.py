"""State definitions.

State is the interface between the graph and end user as well as the
data model used internally by the graph.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class DocumentInfo:
    """Basic document metadata and content."""
    content: Optional[str] = None              # Full extracted text
    path: Optional[str] = None                 # File path
    file_type: Optional[str] = None            # PDF, DOCX, etc.
    title: Optional[str] = None                # Document title
    authors: Optional[List[str]] = None        # Author list
    publication_date: Optional[str] = None     # When published
    document_type: Optional[str] = None        # Report, paper, study, etc.
    page_count: Optional[int] = None           # Number of pages


@dataclass(kw_only=True)
class DocumentStructure:
    """Document structure and formatting details."""
    # Core structure
    sections: Optional[List[dict]] = None      # [{title, level, content, start_page, word_count}]
    table_of_contents: Optional[List[dict]] = None  # [{title, page, level}]
    
    # Special sections
    abstract: Optional[str] = None             # Executive summary/abstract
    introduction: Optional[str] = None         # Introduction section
    conclusion: Optional[str] = None           # Conclusions/summary
    appendix: Optional[List[dict]] = None      # [{title, content, tables, figures}]
    
    # Tables and data
    tables: Optional[List[dict]] = None        # [{title, headers, data, page, section}]
    figures: Optional[List[dict]] = None       # [{title, caption, page, section}]
    
    # References & citations
    references: Optional[List[dict]] = None    # [{text, url, type, page, section}]
    bibliography: Optional[str] = None         # Full bibliography section
    
    # Format characteristics
    formatting_style: Optional[dict] = None   # {heading_styles, fonts, spacing}
    citation_style: Optional[str] = None      # APA, MLA, Chicago, etc.
    data_depth: Optional[dict] = None          # {section_detail_levels, table_complexity}


@dataclass(kw_only=True)
class AnalyticalFramework:
    """Content analysis and argumentation structure."""
    # Content analysis
    main_topics: Optional[List[str]] = None           # Primary topics/themes
    key_questions: Optional[List[str]] = None         # Research questions addressed
    methodology: Optional[str] = None                 # Research methodology used
    data_sources: Optional[List[str]] = None          # Types of data referenced
    
    # Argumentation structure
    thesis_statement: Optional[str] = None            # Main argument/thesis
    supporting_arguments: Optional[List[str]] = None  # Key supporting points
    evidence_types: Optional[List[str]] = None        # Statistical, anecdotal, etc.
    
    # Analysis depth and requirements
    detail_requirements: Optional[dict] = None        # {section: depth_level, data_granularity}
    research_scope: Optional[List[str]] = None        # Geographic, temporal, topical scope


@dataclass(kw_only=True)
class ResearchResults:
    """Research findings and updated sources."""
    new_sources: Optional[List[dict]] = None          # [{url, title, date, relevance, authority}]
    emerging_topics: Optional[List[str]] = None       # New developments since original
    updated_data: Optional[List[dict]] = None         # [{topic, old_value, new_value, source}]
    knowledge_bank_results: Optional[List[dict]] = None  # RAG query results
    source_quality_scores: Optional[dict] = None      # {source_id: quality_score}


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    topic: str
    "The topic for which the agent is tasked to gather information."

    extraction_schema: dict[str, Any]
    "The json schema defines the information the agent is tasked with filling out."

    info: Optional[dict[str, Any]] = field(default=None)
    "The info state tracks the current extracted data for the given topic, conforming to the provided schema. This is primarily populated by the agent."
    
    # NEW: Document input
    document_path: Optional[str] = field(default=None)
    "Path to the document to be analyzed and updated."


@dataclass(kw_only=True)
class State(InputState):
    """A graph's State defines three main things.

    1. The structure of the data to be passed between nodes (which "channels" to read from/write to and their types)
    2. Default values for each field
    3. Reducers for the state's fields. Reducers are functions that determine how to apply updates to the state.
    See [Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) for more information.
    """

    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    """
    Messages track the primary execution state of the agent.

    Typically accumulates a pattern of:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
        information
    3. ToolMessage(s) - the responses (or errors) from the executed tools

        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )

    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`.
        """

    loop_step: Annotated[int, operator.add] = field(default=0)

    # NEW: Document understanding components
    document_info: Optional[DocumentInfo] = field(default=None)
    "Basic document metadata and extracted content."
    
    document_structure: Optional[DocumentStructure] = field(default=None)
    "Document structure including sections, tables, references, and formatting."
    
    analytical_framework: Optional[AnalyticalFramework] = field(default=None)
    "Content analysis including topics, methodology, and argumentation structure."
    
    research_results: Optional[ResearchResults] = field(default=None)
    "Research findings including new sources, updated data, and quality scores."
    
    # NEW: Processing status and quick access
    processing_stage: str = field(default="start")
    "Current processing stage: start, parsed, analyzed, researched, generated."
    
    content_map: Optional[dict] = field(default=None)
    "Quick access map to key sections and data for efficient processing."


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """
