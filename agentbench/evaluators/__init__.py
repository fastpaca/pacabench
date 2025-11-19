from agentbench.config import EvaluatorConfig
from agentbench.evaluators.base import BaseEvaluator
from agentbench.evaluators.judge import LLMJudgeEvaluator
from agentbench.evaluators.matchers import ExactMatchEvaluator, F1Evaluator


def get_evaluator(config: EvaluatorConfig) -> BaseEvaluator:
    if config.type == "exact_match":
        return ExactMatchEvaluator(config)
    elif config.type == "f1":
        return F1Evaluator(config)
    elif config.type == "llm_judge":
        return LLMJudgeEvaluator(config)
    else:
        raise ValueError(f"Unknown evaluator type: {config.type}")
