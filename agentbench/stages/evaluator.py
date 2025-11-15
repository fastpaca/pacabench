"""Stage 3: Evaluation - Evaluators for different task types."""

from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from agentbench.stages.case import Case
from agentbench.stages.runner import RunnerOutput


@dataclass
class EvaluationOutput:
    """Standardized output from all evaluators."""

    passed: bool
    f1_score: float | None = None
    f1_passed: bool | None = None
    judge_passed: bool | None = None
    judge_metrics: dict[str, int] | None = None


def evaluate_multiple_choice(
    case: Case,
    runner_output: RunnerOutput,
) -> EvaluationOutput:
    """
    Evaluate multiple choice answer by comparing first letter.

    Args:
        case: Test case
        runner_output: Runner output

    Returns:
        EvaluationOutput with passed status
    """
    output = runner_output.result
    expected = case.expected_output

    if not output or not expected:
        return EvaluationOutput(passed=False)

    response = output.strip().upper()
    choice = response[0] if response else ""
    expected_choice = expected.strip().upper()

    passed = choice == expected_choice
    return EvaluationOutput(passed=passed, f1_passed=passed)


def evaluate_f1_score(
    case: Case,
    runner_output: RunnerOutput,
) -> EvaluationOutput:
    """
    Evaluate using F1 score based on token overlap.

    Args:
        case: Test case
        runner_output: Runner output

    Returns:
        EvaluationOutput with F1 score and pass status
    """
    output = runner_output.result
    expected = case.expected_output

    if not output or not expected:
        return EvaluationOutput(passed=False, f1_score=0.0, f1_passed=False)

    encoding = tiktoken.get_encoding("cl100k_base")
    response_tokens = set(encoding.encode(output.strip().lower()))
    expected_tokens = set(encoding.encode(expected.strip().lower()))

    if not expected_tokens:
        return EvaluationOutput(passed=False, f1_score=0.0, f1_passed=False)

    overlap = response_tokens & expected_tokens
    if not overlap:
        return EvaluationOutput(passed=False, f1_score=0.0, f1_passed=False)

    precision = len(overlap) / len(response_tokens) if response_tokens else 0.0
    recall = len(overlap) / len(expected_tokens) if expected_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    f1_passed = f1 > 0.5
    return EvaluationOutput(passed=f1_passed, f1_score=f1, f1_passed=f1_passed)


def evaluate_llm_judge(
    case: Case,
    runner_output: RunnerOutput,
    model: str = "gpt-4o-mini",
    openai_client: OpenAI | None = None,
) -> EvaluationOutput:
    """
    Evaluate using LLM-as-judge for semantic equivalence.

    Args:
        case: Test case
        runner_output: Runner output
        model: Judge model to use
        openai_client: OpenAI client (creates new one if None)

    Returns:
        EvaluationOutput with judge result and usage metrics
    """
    output = runner_output.result
    expected = case.expected_output

    if not output or not expected:
        return EvaluationOutput(
            passed=False,
            judge_passed=False,
            judge_metrics={"input_tokens": 0, "output_tokens": 0},
        )

    response = output.strip()
    expected_text = expected.strip()
    question = case.inputs.get("question", "N/A")

    prompt = f"""You are evaluating if a model's answer is semantically equivalent to the expected answer.

Question: {question}

Expected Answer: {expected_text}

Model's Answer: {response}

Does the model's answer convey the same information as the expected answer? Consider:
- Paraphrasing is acceptable
- Extra explanation is acceptable if core answer is present
- Minor details can differ if main point matches

Respond with ONLY "YES" or "NO"."""

    if openai_client is None:
        openai_client = OpenAI()

    completion = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    judgment = completion.choices[0].message.content or ""
    judgment = judgment.strip().upper()

    usage = {
        "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
        "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
    }

    judge_passed = judgment.startswith("YES")
    return EvaluationOutput(
        passed=judge_passed,
        judge_passed=judge_passed,
        judge_metrics=usage,
    )


def evaluate_gaia(
    case: Case,
    runner_output: RunnerOutput,
    model: str = "gpt-4o-mini",
    openai_client: OpenAI | None = None,
) -> EvaluationOutput:
    """
    Evaluate GAIA answer using LLM-as-judge.

    Args:
        case: Test case
        runner_output: Runner output
        model: Judge model to use
        openai_client: OpenAI client (creates new one if None)

    Returns:
        EvaluationOutput with judge result and usage metrics
    """
    output = runner_output.result
    expected = case.expected_output

    if not output or not expected:
        return EvaluationOutput(
            passed=False,
            judge_passed=False,
            judge_metrics={"input_tokens": 0, "output_tokens": 0},
        )

    response = output.strip()
    expected_text = expected.strip()
    question = case.inputs.get("question", "N/A")

    prompt = f"""You are evaluating if an AI assistant's answer matches the expected answer for a GAIA benchmark question.

Question: {question}

Expected Answer: {expected_text}

Assistant's Answer: {response}

Does the assistant's answer match the expected answer? Consider:
- The core factual content must be correct
- Paraphrasing is acceptable
- Extra context/explanation is acceptable if core answer is present
- Minor formatting differences are acceptable (e.g., "50%" vs "50 percent")
- But factual errors or missing key information mean NO match

Respond with ONLY "YES" or "NO"."""

    if openai_client is None:
        openai_client = OpenAI()

    completion = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    judgment = completion.choices[0].message.content or ""
    judgment = judgment.strip().upper()

    usage = {
        "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
        "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
    }

    judge_passed = judgment.startswith("YES")
    return EvaluationOutput(
        passed=judge_passed,
        judge_passed=judge_passed,
        judge_metrics=usage,
    )
