"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from enrichment_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    


    vision_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-haiku-20241022",
        metadata={
            "description": "The name of the vision-capable language model to use for visual document analysis like title and table of contents extraction. "
            "Should be in the form: provider/model-name. Supports anthropic/claude-3-5-haiku-20241022 or openai/gpt-4o-mini."
        },
    )

    bibliography_vision_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the vision-capable language model to use specifically for bibliography extraction. "
            "Uses a more powerful model to handle complex bibliography parsing and avoid truncation. "
            "Should be in the form: provider/model-name. Supports anthropic/claude-3-5-sonnet-20240620 or openai/gpt-4o."
        },
    )

    prompt: str = field(
        default=prompts.MAIN_PROMPT,
        metadata={
            "description": "The main prompt template to use for the agent's interactions. "
            "Expects two f-string arguments: {info} and {topic}."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    max_info_tool_calls: int = field(
        default=3,
        metadata={
            "description": "The maximum number of times the Info tool can be called during a single interaction."
        },
    )

    max_loops: int = field(
        default=6,
        metadata={
            "description": "The maximum number of interaction loops allowed before the agent terminates."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Load configuration w/ defaults for the given invocation."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
