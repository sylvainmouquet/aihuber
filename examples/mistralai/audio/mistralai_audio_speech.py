import os

from dotenv import load_dotenv

from aihuber import LLM
from pathlib import Path


load_dotenv()

if __name__ == "__main__":
    """Audio example"""
    llm = LLM(
        model="mistral:mistral-small-latest", api_key=os.getenv("MISTRAL_AI_TOKEN")
    )

    speech_file_path = Path(__file__).parent / "speech.mp3"
    with llm.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input="The quick brown fox jumped over the lazy dog.",
    ) as response:
        response.stream_to_file(speech_file_path)
