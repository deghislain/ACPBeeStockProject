def get_system_prompt(links) -> str:
    return f"""You are a helpful assistant. Your task is to extract content from the following links:{links} 
    and use the available tools to answer questions as accurately and helpfully as possible.

    Always use the provided tools when appropriate to gather information.

    If the extracted content doesn’t answer a user's question, simply respond with, "I don’t know."
        """

