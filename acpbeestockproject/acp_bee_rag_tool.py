import re
import sys
from typing import Any
from typing import List
from typing import Set
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from agents import FunctionTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import MistralAIEmbeddings

sys.path.insert(0, '../')
from utils import load_config

config = load_config()
dep_config = config["deployment"]


def get_valid_urls(links: List[str]) -> Set[str]:
    """
    Extract and validate URLs from a given string.

    Args:
        links (str): The text containing potential URLs.

    Returns:
        Set[str]: A set of valid URLs found in the input string.
    """

    # Regular expression to match URLs
    url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Find all occurrences of URLs in the input string
    potential_urls = url_regex.findall(links)

    # Set to store valid URLs
    valid_urls = set()

    # Process each potential URL
    for url in potential_urls:
        try:
            # Remove any trailing commas or closing braces that might contaminate URLs
            cleaned_url = url.strip().replace('],', '').replace(']', '')
            valid_urls.add(cleaned_url)
        except Exception as ex:
            # Log the exception and continue processing other URLs
            print(f"Error while processing URL: {ex}")

    return valid_urls


def extract_page_content(urls: Set[str]):
    """
        Extract content from web pages given a set of URLs.

        Args:
            urls (List[str]): List of URLs to scrape the content from.

        Returns:
            List[str]: A list containing the combined content from all web pages.
        """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def store_page_content_in_vector_db(contents: List[str]) -> Chroma:
    logging.info(f"store_page_content_in_vector_db************************* START: contents = {contents}")
    """
    Store page contents in a vector database using the Chroma library.

    Args:
        contents (str): The text content to process and store.

    Returns:
        Chroma: An instance of Chroma with documents indexed and embeddings computed.
    """
    # Initialize text splitter with specified chunk size and no overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    # Split documents based on the provided content
    content_splits = text_splitter.split_documents(contents)

    # Initialize embeddings with Mistral model and API key
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=dep_config["MISTRAL_API_KEY"]
    )

    # Create Chroma instance to store documents and compute embeddings
    return Chroma.from_documents(
        documents=content_splits,
        collection_name="agentic-rag-chroma",
        embedding=embeddings,
    )


def get_retriever(links: List[str]):
    logging.info(f"get_retriever************************* START: links = {links}")
    """
            Creates a retriever using the content of website.
            Args:
                 links: urls which content is necessary to create the retriever

            Returns:
                Returns a retriever
                  """
    urls = get_valid_urls(links)
    pages_content = extract_page_content(urls)
    return store_page_content_in_vector_db(pages_content).as_retriever()


class RAGTool(FunctionTool):
    name = "RagTool"
    description = "It Create a retriever using the content of a website"

    def __init__(self, params_json_schema: dict[str, Any] | None = None) -> None:
        self.params_json_schema = params_json_schema

    def on_invoke_tool(self, context: Any, question: str) -> str:
        logging.info(f"on_invoke_tool************************* START: links = {question}")
        """
        Creates a retriever using the content of a website and use  the retriever
        to returns the answer to user's question.

        Args:
            context (Any): The tool run context.
            question (str): The question to be answered.

        Returns:
            str: The answer retrieved from the documents.
        """
        retriever_result = get_retriever(self.params_json_schema["links"]).invoke(question)
        final_answer = "\n".join(retriever_result)

        return final_answer
