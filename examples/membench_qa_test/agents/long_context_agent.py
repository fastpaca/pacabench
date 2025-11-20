import json
import os
import sys

from openai import OpenAI


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        question = data.input
        history = data.history
        system_prompt = "You are a helpful assistant with a long memory."
        user_prompt = f"History:\n{history}\n\nQuestion: {question}"
        choices = data.get("choices")
        if choices and isinstance(choices, dict):
            choices_text = "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
            user_prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            output = response.choices[0].message.content

            print(json.dumps({"output": output, "error": None}))
        except Exception as e:
            print(json.dumps({"output": None, "error": str(e)}))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
