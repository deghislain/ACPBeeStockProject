from typing import List
from acp.server.highlevel import Server, Context
from beeai_sdk.providers.agent import run_agent_provider
from beeai_sdk.schemas.text import TextInput, TextOutput
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
import traceback
from beeai_framework.errors import FrameworkError
import streamlit as st
import logging
import sys
import asyncio
from acp_bee_rag_tool import RAGTool

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


async def run(links: List[str], input: str):
    server = Server("rag-agent")

    logging.info(f"Responding to the following question: {input} ")

    @server.agent(
        name="rag-agent",
        description="Rag Agent to showcase beeai platform extension",
        input=TextInput,
        output=TextOutput
    )
    async def generate_response(input: TextInput, ctx: Context) -> TextOutput:
        logging.info(f"Generate response start with input : {input} ")
        model = OpenAIChatCompletionsModel(
            model="ollama:granite3-dense:latest",
            openai_client=AsyncOpenAI(base_url="http://localhost:11434")
        )
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            tools=[RAGTool({"links": links})],
            model=model
        )
        result = await Runner.run(agent, f"Provide an answer for this question {input}")
        logging.info(f"Model response ***********************result : {result.final_output} ")
        return TextOutput(result=result.final_output)

    await run_agent_provider(server)


def main(input: List[str], question: str):
    try:
        response = asyncio.run(run(input, question))
        logging.info(f"Model response *********************** : {response} ")
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())



if __name__ == "__main__":
    links = st.text_area(":blue[Add the sources(URL) for your Q&A session]", placeholder="Paste your links here")
    if st.button("Submit"):
        st.session_state['links'] = links
        st.chat_input(placeholder="Type your question")
    elif "links" in st.session_state:
        links = st.session_state['links']
        if links:
            question = st.chat_input(placeholder="Type your question")
            if question:
                main(links, question)

