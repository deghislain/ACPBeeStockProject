from acp.server.highlevel import Server, Context
from beeai_sdk.providers.agent import run_agent_provider
from beeai_sdk.schemas.metadata import UiDefinition, UiType
from beeai_sdk.schemas.text import TextInput, TextOutput
import logging

from agents import Agent, Runner
import asyncio
from acp_bee_rag_tool import RAGTool


async def run():
    server = Server("rag-agent")

    @server.agent(
        name="rag-agent",
        description="Rag Agent to showcase beeai platform extension",
        input=TextInput,
        output=TextOutput,
        ui=UiDefinition(type=UiType.hands_off, userGreeting="Write a haiku poem about:")
    )
    async def generate_response(input: TextInput, ctx: Context) -> TextOutput:
        urls = []
        llm = "ollama:granite3-dense:latest"
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            tools=[RAGTool({"links": urls})],
            model=llm
        )
        result = await Runner.run(agent, f"Write a haiku about {input}")
        return TextOutput(result=result)

    await run_agent_provider(server)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
