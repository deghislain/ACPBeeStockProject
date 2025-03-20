from acp.server.highlevel import Server, Context
from beeai_sdk.providers.agent import run_agent_provider
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
import traceback
from beeai_framework.errors import FrameworkError
from acp_bee_rag_io import RagInput, RagOutput
import logging
import sys
import asyncio
from acp_bee_rag_tool import RAGTool

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


async def run():
    server = Server("rag-agent")

    @server.agent(
        name="rag-agent",
        description="Rag Agent to showcase beeai platform extension",
        input=RagInput,
        output=RagOutput
    )
    async def generate_response(input: RagInput, ctx: Context) -> RagOutput:
        logging.info(f"Generate response START: input= {input} ")

        model = OpenAIChatCompletionsModel(
            model="granite3-dense:latest",
            openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        )
        links = input.input["links"]
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            tools=[RAGTool({"links": links})],
            model=model
        )
        question = input.input["text"]
        result = await Runner.run(agent, f"Answer the following question: {question}")
        response = {"output": result.final_output}
        rag_output = RagOutput(output=response)
        logging.info(f"Generate response completed: output= {response}")
        return rag_output

    try:
        await run_agent_provider(server)
    except Exception as e:
        logging.error(f"Error running agent provider: {e}")


def main():
    try:
        asyncio.run(run())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())


if __name__ == "__main__":
    main()
