"""Evaluators for different task types."""

from typing import Any

import tiktoken
from openai import OpenAI


def evaluate_multiple_choice(
    output: str | None,
    expected: str,
    inputs: dict[str, Any] | None = None,
) -> bool:
    """
    Evaluate multiple choice answer by comparing first letter.

    Args:
        output: Model output
        expected: Expected answer
        inputs: Optional inputs (unused)

    Returns:
        True if first letter matches expected
    """
    if not output or not expected:
        return False

    response = output.strip().upper()
    choice = response[0] if response else ""
    expected_choice = expected.strip().upper()

    return choice == expected_choice


def evaluate_f1_score(
    output: str | None,
    expected: str,
    inputs: dict[str, Any] | None = None,
) -> tuple[bool, float]:
    """
    Evaluate using F1 score based on token overlap.

    Args:
        output: Model output
        expected: Expected answer
        inputs: Optional inputs (unused)

    Returns:
        Tuple of (pass: True if F1 > 0.5, f1_score: float)
    """
    if not output or not expected:
        return False, 0.0

    encoding = tiktoken.get_encoding("cl100k_base")
    response_tokens = set(encoding.encode(output.strip().lower()))
    expected_tokens = set(encoding.encode(expected.strip().lower()))

    if not expected_tokens:
        return False, 0.0

    overlap = response_tokens & expected_tokens
    if not overlap:
        return False, 0.0

    precision = len(overlap) / len(response_tokens) if response_tokens else 0.0
    recall = len(overlap) / len(expected_tokens) if expected_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1 > 0.5, f1


def evaluate_llm_judge(
    output: str | None,
    expected: str,
    inputs: dict[str, Any],
    model: str = "gpt-4o-mini",
    openai_client: OpenAI | None = None,
) -> tuple[bool, dict[str, int]]:
    """
    Evaluate using LLM-as-judge for semantic equivalence.

    Args:
        output: Model output
        expected: Expected answer
        inputs: Case inputs (must contain "question")
        model: Judge model to use
        openai_client: OpenAI client (creates new one if None)

    Returns:
        Tuple of (pass: bool, usage: dict with input/output tokens)
    """
    if not output or not expected:
        return False, {"input_tokens": 0, "output_tokens": 0}

    response = output.strip()
    expected_text = expected.strip()
    question = inputs.get("question", "N/A")

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

    return judgment.startswith("YES"), usage


def evaluate_gaia(
    output: str | None,
    expected: str,
    inputs: dict[str, Any],
    model: str = "gpt-4o-mini",
    openai_client: OpenAI | None = None,
) -> tuple[bool, dict[str, int]]:
    """
    Evaluate GAIA answer using LLM-as-judge.

    Args:
        output: Model output
        expected: Expected answer
        inputs: Case inputs (must contain "question")
        model: Judge model to use
        openai_client: OpenAI client (creates new one if None)

    Returns:
        Tuple of (pass: bool, usage: dict with input/output tokens)
    """
    if not output or not expected:
        return False, {"input_tokens": 0, "output_tokens": 0}

    response = output.strip()
    expected_text = expected.strip()
    question = inputs.get("question", "N/A")

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

    return judgment.startswith("YES"), usage
