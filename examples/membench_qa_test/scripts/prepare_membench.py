import os
import json
import glob
from pathlib import Path

def main():
    dataset_path = os.environ.get("AGENTBENCH_DATASET_PATH")
    if not dataset_path:
        print("AGENTBENCH_DATASET_PATH not set, assuming current directory or skipping.")
        return

    base_path = Path(dataset_path)
    # MemBench structure: MemData/FirstAgent/*.json
    source_dir = base_path / "MemData" / "FirstAgent"
    output_file = base_path / "membench.jsonl"
    
    print(f"Converting files from {source_dir} to {output_file}")
    
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    with open(output_file, "w") as outfile:
        for json_file in glob.glob(str(source_dir / "*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                # Each file might contain multiple events/cases?
                # Based on old membench.py:
                # records = data.get("events") or data.get("multi_agent") or []
                
                records = data.get("events") or data.get("multi_agent") or []
                for record in records:
                    # Flatten
                    tid = record.get("tid", "unknown")
                    qa = record.get("QA", {})
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")
                    ground_truth = qa.get("ground_truth", "")
                    
                    # We need to construct a history string or list?
                    # Case inputs usually are single string input.
                    # Or we can pass metadata.
                    # But standard AgentBench agents usually expect 'input'.
                    # The 'history' field in Case can be populated.
                    # But our datasets/git.py just maps input_map keys.
                    
                    # If we want to pass conversation history, we need to format it into 'input'
                    # or rely on 'history' field if our prepare script handles it.
                    # But GitDataset loader is generic.
                    # It reads `input` key from JSONL.
                    
                    # Let's convert message_list to string representation for `input`?
                    # Or just pass question as input and history in history field.
                    # GitDataset doesn't explicitly map 'history'.
                    # It loads everything into metadata.
                    # So agents should read 'metadata' or we should format 'input' to include history.
                    
                    # For long-context agent, input should be full context.
                    # For mem0 agent, maybe it handles history differently.
                    
                    # Let's format the conversation into a single text block for 'input' 
                    # if the agent expects a single prompt.
                    # But wait, 'question' is the query. 'message_list' is context.
                    
                    message_list = record.get("message_list", [])
                    context = []
                    for msg in message_list:
                         if isinstance(msg, dict):
                             u = msg.get("user_message") or msg.get("user")
                             a = msg.get("assistant_message") or msg.get("assistant")
                             if u: context.append(f"User: {u}")
                             if a: context.append(f"Assistant: {a}")
                    
                    history_text = "\n".join(context)
                    
                    # For now, let's just put the question as input, 
                    # and maybe putting history in metadata is enough.
                    # BUT the generic CommandRunner passes `input` and `history` and `metadata`.
                    # GitDataset maps `input` from JSONL.
                    # If we want `history` in Case, GitDataset needs to support it.
                    # I didn't implement `history` mapping in `GitDataset`.
                    # I only implemented `input_map` for input/expected.
                    
                    # However, `LocalDataset` and `GitDataset` load the whole JSON object into `metadata`.
                    # `CommandRunner` passes `case.metadata` into the input JSON sent to agent.
                    # So the agent receives everything in the JSONL line.
                    
                    # So we can put `history` in the JSONL.
                    
                    out_record = {
                        "case_id": f"{Path(json_file).stem}-{tid}",
                        "question": question,
                        "ground_truth": str(ground_truth),
                        "history": message_list, # Raw list
                        "context_text": history_text
                    }
                    
                    outfile.write(json.dumps(out_record) + "\n")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    main()

