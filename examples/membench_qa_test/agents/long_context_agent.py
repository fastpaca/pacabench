import json
import os
import sys
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InputCase(BaseModel):
    case_id: str
    input: str
    history: list[Message] = Field(default_factory=list)
    choices: dict[str, str] = Field(default_factory=dict)


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case = InputCase.model_validate(data)
        except Exception:
            # Skip malformed lines
            continue

        system_prompt = "You are a helpful assistant with a long memory."
        messages = [{"role": "system", "content": system_prompt}]

        messages.extend([m.model_dump() for m in case.history])

        user_prompt = f"Question: {case.input}"
        if case.choices:
            choices_text = "\n".join(
                f"{key}. {value}" for key, value in sorted(case.choices.items())
            )
            user_prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            output = response.choices[0].message.content
            print(json.dumps({"output": output, "error": None}))
        except Exception as e:
            print(json.dumps({"output": None, "error": str(e)}))

        sys.stdout.flush()


if __name__ == "__main__":
    main()
