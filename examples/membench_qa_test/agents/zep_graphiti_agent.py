import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import Literal

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from openai import OpenAI
from pydantic import BaseModel, Field

# Initialize logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("zep_agent")
logger.setLevel(logging.DEBUG)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("graphiti_core").setLevel(logging.INFO)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: str | None = None


class InputCase(BaseModel):
    case_id: str
    input: str
    history: list[Message] = Field(default_factory=list)
    choices: dict[str, str] = Field(default_factory=dict)


async def process_case(line: str, graphiti: Graphiti, client: OpenAI, model: str):
    try:
        data = json.loads(line)
        case = InputCase.model_validate(data)
    except Exception:
        return

    logger.info(f"Processing case: {case.case_id}")

    group_id = str(case.case_id)

    try:
        # History is already normalized
        await graphiti.add_episode_bulk(
            bulk_episodes=[
                RawEpisode(
                    name=f"History {i}",
                    content=f"{msg.role}: {msg.content}",
                    source=EpisodeType.message,
                    source_description=f"History message from {msg.role}",
                    reference_time=datetime.now(UTC),
                )
                for i, msg in enumerate(case.history)
            ],
            group_id=group_id,
        )

        results = await graphiti.search(query=case.input, group_ids=[group_id], num_results=10)
        memory_context = ""
        if results:
            lines = []
            for edge in results:
                fact = getattr(edge, "fact", str(edge))
                lines.append(f"- {fact}")
            if lines:
                memory_context = (
                    "Relevant context from knowledge graph:\n" + "\n".join(lines) + "\n\n"
                )

        prompt_content = f"{memory_context}Question: {case.input}"
        system_prompt = "You are a helpful assistant with a long memory."

        if case.choices:
            choices_text = "\n".join(
                f"{key}. {value}" for key, value in sorted(case.choices.items())
            )
            prompt_content += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

        # Run sync OpenAI call in executor to avoid blocking async loop
        loop = asyncio.get_running_loop()

        def call_openai():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content},
                ],
            )
            return response.choices[0].message.content

        output = await loop.run_in_executor(None, call_openai)

        print(json.dumps({"output": output, "error": None}))
        sys.stdout.flush()
        logger.info(f"Finished case {case.case_id}")

    except Exception as e:
        logger.exception(f"Error in agent loop: {e}")
        print(json.dumps({"output": None, "error": str(e)}))
        sys.stdout.flush()


async def run_agent():
    proxy_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    llm_config = LLMConfig(api_key=api_key, model=model, base_url=proxy_url)
    llm_client = OpenAIClient(config=llm_config)
    embedder_config = OpenAIEmbedderConfig(
        api_key=api_key, base_url=proxy_url, embedding_model="text-embedding-3-small"
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    graphiti = None
    # Retry loop for Neo4j connection
    for i in range(10):
        try:
            graphiti = Graphiti(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
            )
            try:
                await graphiti.build_indices_and_constraints()
            except Exception as e:
                # Ignore if schema rules already exist (common in Neo4j restarts)
                if "EquivalentSchemaRuleAlreadyExists" not in str(e):
                    raise e
            break
        except Exception as e:
            logger.warning(f"Attempt {i + 1}: Failed to init Graphiti: {e}")
            time.sleep(2)

    if not graphiti:
        logger.error("Could not connect to Neo4j after multiple attempts.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=proxy_url)
    logger.info("Agent ready")

    # Standard async stdin reading
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line_bytes = await reader.readline()
            if not line_bytes:
                logger.info("EOF received on stdin")
                break
            line = line_bytes.decode()
            if not line.strip():
                continue
            # Process sequentially to avoid Graphiti concurrency issues
            await process_case(line, graphiti, client, model)
        except Exception as e:
            logger.error(f"Error reading stdin: {e}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except Exception as e:
        sys.stderr.write(f"Fatal error: {e}\n")
