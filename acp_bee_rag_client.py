from typing import List
from acp.server.highlevel import Server, Context
from beeai_sdk.providers.agent import run_agent_provider
from beeai_sdk.schemas.text import TextInput, TextOutput
import streamlit as st


from agents import Agent, Runner
import asyncio
from acp_bee_rag_tool import RAGTool


async def run(links: List[str], input: str):
    server = Server("rag-agent")
    print("***********************************running ", input)
    @server.agent(
        name="rag-agent",
        description="Rag Agent to showcase beeai platform extension",
        input=TextInput,
        output=TextOutput
    )
    async def generate_response(input: TextInput, ctx: Context) -> TextOutput:
        llm = "ollama:granite3-dense:latest"
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            tools=[RAGTool({"links": links})],
            model=llm
        )
        result = await Runner.run(agent, f"Provide an answer for this question {input}")
        return TextOutput(result=result)

    await run_agent_provider(server)


def main(input: List[str], question: str):
    asyncio.run(run(input, question))


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
                print("***********************************question ", question)
                main(links, question)