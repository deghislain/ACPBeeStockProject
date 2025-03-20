import asyncio
from typing import List
from beeai_framework.errors import FrameworkError
from acp import ClientSession
from acp.client.sse import sse_client
import logging
import traceback
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


async def run_client(question: str, links: List[str]):
    logging.info(f"run_client************************* START: question = {question}")
    async with sse_client(url="http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            # List agents
            agents = await session.list_agents()
            print("********************Available agents:", [agent.name for agent in agents.agents])
            # Run agent
            response = await session.run_agent("rag-agent", {"input": {"text": question, "links": links}})
            for data in response:
                if data[1] != None:
                    response = data[1]["output"]["output"]
                    print(response)
            return response


def main(question: str, links: List[str]):
    try:
        asyncio.run(run_client(question, links))
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())


if __name__ == "__main__":
    links = ["https://developer.nvidia.com/agentiq"]
    question = "what is an agent?"
    asyncio.run(run_client(question, links))
