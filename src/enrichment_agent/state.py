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
    path: Optional[str] = None                 # File path
    file_type: Optional[str] = None            # PDF, DOCX, etc.
    title: Optional[str] = None                # Document title
    publication_date: Optional[str] = None     # When published
    page_count: Optional[int] = None           # Number of pages
    metadata: Optional[dict] = None            # Extracted metadata from user query


@dataclass(kw_only=True)
class DocumentStructure:
    """Document structure and formatting details."""
    table_of_contents: Optional[List[dict]] = None  # [{title, page, level}]
    references: Optional[List[dict]] = None    # [{name, year, link}]
    tables: Optional[dict] = None              # {page_num: [table_data]}


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    topic: str
    "The topic for which the agent is tasked to gather information."

    extraction_schema: dict[str, Any]
    "The json schema defines the information the agent is tasked with filling out."

    info: Optional[dict[str, Any]] = field(default=None)
    "The info state tracks the current extracted data for the given topic, conforming to the provided schema. This is primarily populated by the agent."
    
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

    # Document understanding components
    document_info: Optional[DocumentInfo] = field(default=None)
    "Basic document metadata and extracted content."
    
    document_structure: Optional[DocumentStructure] = field(default=None)
    "Document structure including TOC, references, and tables."
    
    # Processing status
    processing_stage: str = field(default="start")
    "Current processing stage: start, document_analysis_complete, etc."


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
