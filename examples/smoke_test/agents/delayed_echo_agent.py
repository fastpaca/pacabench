"""Slow echo agent for the kill/resume demo.

Same protocol as echo_agent.py, with a per-case sleep so a kill can land
between cases. PACABENCH_DEMO_DELAY_SECONDS sets the sleep (default 1).
The offline smoke config (pacabench.yaml) does not use this agent.
"""

import json
import os
import sys
import time


def main():
    delay = float(os.environ.get("PACABENCH_DEMO_DELAY_SECONDS", "1"))
    for line in sys.stdin:
        if not line:
            continue
        try:
            data = json.loads(line)
            time.sleep(delay)
            response = data["input"][::-1]
            print(json.dumps({"output": response}))
            sys.stdout.flush()
        except json.JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
