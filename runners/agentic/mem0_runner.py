#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from mem0 import AsyncMemory
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, PythonInterpreterTool, VisitWebpageTool, WikipediaSearchTool
from smolagents.models import OpenAIServerModel

async def main_async():
    case = json.loads(sys.stdin.read())
    model = os.getenv("MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    config = {"vector_store": {"provider": "qdrant", "config": {"collection_name": "agentbench_gaia", "host": "memory"}}, "embedder": {"provider": "openai"}}
    memory = await AsyncMemory.from_config(config)
    model_obj = OpenAIServerModel(model_id=model, api_base=base_url, api_key=api_key)
    tools = [DuckDuckGoSearchTool(), WikipediaSearchTool(), VisitWebpageTool(), PythonInterpreterTool(), FinalAnswerTool()]
    agent = CodeAgent(tools=tools, model=model_obj, max_steps=15, verbosity_level=0)
    
    user_id = case["id"]
    question = case["inputs"]["question"]
    
    search_result = await memory.search(query=question, user_id=user_id, limit=5)
    relevant_memories = search_result.get("results", []) if isinstance(search_result, dict) else (search_result if isinstance(search_result, list) else [])
    
    augmented_question = question
    if relevant_memories:
        memory_lines = []
        for idx, mem in enumerate(relevant_memories, 1):
            memory_text = mem.get("memory", "") or mem.get("text", "") if isinstance(mem, dict) else (mem if isinstance(mem, str) else str(mem))
            if memory_text:
                memory_lines.append(f"{idx}. {memory_text}")
        if memory_lines:
            memory_context = "Relevant context from previous tasks:\n" + "\n".join(memory_lines) + "\n\n"
            augmented_question = memory_context + question
    
    if case["inputs"].get("file_name"):
        augmented_question += f"\n\nNote: A file '{case['inputs']['file_name']}' is mentioned but not yet accessible."
    
    try:
        result = agent.run(augmented_question, return_full_result=True)
        answer = str(result.output).strip()
        await memory.add([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], user_id=user_id)
        print(json.dumps({"result": answer, "error": None}))
    except Exception as e:
        print(json.dumps({"result": None, "error": str(e)}))
        sys.exit(1)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
