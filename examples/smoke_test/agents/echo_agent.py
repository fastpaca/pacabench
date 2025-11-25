import json
import sys


def main():
    for line in sys.stdin:
        if not line:
            continue
        try:
            data = json.loads(line)
            # Echo agent just returns the input reversed
            response = data["input"][::-1]

            output = {
                "output": response,
                "metrics": {
                    "call_count": 0,
                    "input_tokens": len(data["input"]),
                    "output_tokens": len(response),
                    "latency_ms": 10,
                },
            }
            print(json.dumps(output))
            sys.stdout.flush()
        except json.JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
