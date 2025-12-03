"""Evaluators for PacaBench."""

from pacabench.evaluators.base import BaseEvaluator
from pacabench.evaluators.judge import LLMJudgeEvaluator
from pacabench.evaluators.matchers import ExactMatchEvaluator, F1Evaluator, MultipleChoiceEvaluator
from pacabench.models import EvaluatorConfig


def get_evaluator(config: EvaluatorConfig) -> BaseEvaluator:
    """Create an evaluator based on the config type."""
    if config.type == "exact_match":
        return ExactMatchEvaluator(config)
    elif config.type == "f1":
        return F1Evaluator(config)
    elif config.type == "llm_judge":
        return LLMJudgeEvaluator(config)
    elif config.type == "multiple_choice":
        return MultipleChoiceEvaluator(config)
    else:
        raise ValueError(f"Unknown evaluator type: {config.type}")
