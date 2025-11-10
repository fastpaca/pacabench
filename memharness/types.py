from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationSample:
    """A normalized MemBench conversation plus its QA annotation."""

    id: str
    source_file: str
    category_path: List[str]
    tid: Optional[int | str]
    conversation: List[Dict[str, Any]]
    question: str
    choices: Dict[str, str]
    ground_truth: Optional[str]
    answer_text: Optional[str]
    target_step_id: List[Any]
    timestamp: Optional[str]


@dataclass
class QAInput:
    """Structured payload provided to an Answerer implementation."""

    sample_id: str
    history: List[Dict[str, Any]]
    question: str
    choices: Dict[str, str]
    timestamp: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    """Token/request accounting returned by an Answerer."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AnswerResult:
    """LLM (or rule-based) answer for a QAInput."""

    choice: Optional[str]
    response_text: Optional[str] = None
    usage: Optional[UsageStats] = None
    extra: Dict[str, Any] = field(default_factory=dict)
