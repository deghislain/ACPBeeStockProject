import asyncio
from typing import List
from beeai_framework.errors import FrameworkError
from acp import ClientSession
from acp.client.sse import sse_client
import streamlit as st
import logging
import traceback
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
chat_history = []


def display_chat_history():
    chat_history = st.session_state['chat_history']
    count = 0
    for m in chat_history:
        if count % 2 == 0:
            output = st.chat_message("user")
            output.write(m)
        else:
            output = st.chat_message("assistant")
            output.write(m)
        count += 1


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
        return asyncio.run(run_client(question, links))
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
                response = main(question, links)
                if "chat_history" in st.session_state:
                    chat_history = st.session_state['chat_history']
                chat_history.extend([question, response])
                st.session_state['chat_history'] = chat_history

        display_chat_history()
