"""Two-question QA agent for the Quick Start.

Reads one JSON object per line from stdin and writes one JSON object per line
to stdout. Calls the Chat Completions API. Honors OPENAI_BASE_URL so the
PacaBench proxy can record latency and tokens.

Python 3 standard library only. The process inherits OPENAI_API_KEY from the
shell. Do not set that variable in pacabench.yaml: the harness copies agent
env values through as written, and a "${OPENAI_API_KEY}" placeholder would
replace the real key.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def chat(question: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = base + "/chat/completions"
    body = json.dumps(
        {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Reply with only the short answer. No explanation.",
                },
                {"role": "user", "content": question},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("model returned an empty answer")
    return content.strip()


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            answer = chat(data["input"])
            print(json.dumps({"output": answer}))
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
