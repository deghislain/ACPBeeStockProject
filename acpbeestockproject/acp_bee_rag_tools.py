from agents import function_tool
from acp_bee_rag_tool_utils import get_retriever
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@function_tool
def create_qa_context(links, question) -> str:
    logging.info(f"create_qa_context START: input= {links} ")
    """
        Creates the context for a question and answer session.
        Args:
             links: urls which content is necessary to create the context for a question and answer session
             question: the user question

        Returns:
            str: The answer retrieved from the documents.
              """
    retriever_result = get_retriever(links).invoke(question)
    final_answer = "\n".join(retriever_result)
    return final_answer
