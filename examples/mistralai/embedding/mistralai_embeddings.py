import os

from dotenv import load_dotenv

from aihuber import LLM

load_dotenv()

if __name__ == "__main__":
    """Embeddings example"""
    llm = LLM(
        model="mistral:mistral-small-latest", api_key=os.getenv("MISTRAL_AI_TOKEN")
    )

    response = llm.embeddings.create(
        model="mistral-embed", inputs=["Embed this sentence.", "As well as this one."]
    )
    print(f"Response: {response}")
