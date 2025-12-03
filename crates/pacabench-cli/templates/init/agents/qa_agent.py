#!/usr/bin/env python3
"""
Example QA agent for PacaBench.

This agent reads questions from stdin (JSON lines format) and uses OpenAI
to generate answers. It follows the PacaBench agent protocol:

Input: JSON lines with {"input": "question", ...}
Output: JSON lines with {"output": "answer", "metrics": {...}}
"""

import json
import os
import sys
from openai import OpenAI


def main():
    """Main agent loop."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            '{"error": "OPENAI_API_KEY environment variable not set"}',
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    for line in sys.stdin:
        if not line.strip():
            continue

        try:
            data = json.loads(line)
            question = data.get("input", "")

            if not question:
                print(
                    json.dumps({"error": "Missing 'input' field"}),
                    file=sys.stderr,
                )
                continue

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
            )

            answer = response.choices[0].message.content
            usage = response.usage

            # Output in PacaBench format
            output = {
                "output": answer,
                "metrics": {
                    "call_count": 1,
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "latency_ms": 0,  # Will be measured by PacaBench
                },
            }

            print(json.dumps(output))
            sys.stdout.flush()

        except json.JSONDecodeError as e:
            print(
                json.dumps({"error": f"Invalid JSON: {e}"}),
                file=sys.stderr,
            )
        except Exception as e:
            print(
                json.dumps({"error": f"Agent error: {e}"}),
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
