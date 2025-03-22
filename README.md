Retrieval Augmented Generation Application (RAG) with ACP and BeeAI

This project showcases an innovative AI application that leverages the Agent Communication Protocol (ACP) and the BeeAI framework. It simplifies interactions between tools, agents, and existing systems, enabling seamless integration for enhanced functionality.
Overview

This application is a Retrieval Augmented Generation (RAG) system designed to retrieve content from a provided list of URLs and facilitate the questioning of that content through an interactive chat interface.
Architecture

The application comprises three major components:

    Client Interface (acp_bee_rag_client): User interaction hub, allowing users to engage with the application and query the content.

    Server and Content Extraction (acp_bee_rag_server): This pivotal component includes tools and agents that extract content from websites, storing it in a ChromaDB for efficient search and retrieval.

    Content Scraping and Semantic Search Tools (acp_bee_rag_tools): Utilizes advanced scraping techniques and integrates with ollama (powered by granite3-dense:latest) to enable semantic search capabilities across the collected data.

Technologies Employed

    ChromaDB: The vector database that stores the content from web pages, optimizing search and retrieval processes.

    BeeAI: Facilitates the discovery, execution, and composition of AI agents, enhancing the application's adaptive and intelligent nature.

    Agent Communication Protocol (ACP): Enables seamless communication and coordination between different software components.

    Streamlit: Powering the intuitive and responsive graphical user interface (GUI) of the client.

    ollama with granite3-dense:latest: Provides strong language processing capabilities for semantic search and understanding within the content corpus.

Setup & Running the Application

    Initialize the Server:
    Execute the following command in the ACPBeeStockProject/acpbeestockproject directory:

    uv run acp_bee_rag_server.py
    

    Launch the Client Interface:
    After the server is running, open another terminal and navigate back to the project directory, then run:

    uv run streamlit run acp_bee_rag_client.py
    

    Interact with the Application:
        Open your web browser and visit the provided URL (usually http://localhost:8501 by default).
        Input your list of URLs in the designated text area.
        Click 'Submit' to initiate content retrieval and indexation.
        Once ready, start your chat to query the newly indexed content.
