import logging
import numpy as np
from typing import Optional, BinaryIO
from scipy.io import wavfile
from openai import AsyncOpenAI
import io
import os
import asyncio

logger = logging.getLogger(__name__)

class VoiceAgent:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def process_audio_file(self, audio_file: BinaryIO) -> Optional[str]:
        if await self.is_silence(audio_file):
            logger.info("File contains only silence")
            return None

        return await self.transcribe_audio(audio_file)

    async def is_silence(self, audio_file: BinaryIO, max_amplitude_threshold: int = 3000) -> bool:
        try:
            original_position = audio_file.tell()
            audio_file.seek(0)
            
            # Read the WAV file data
            with io.BytesIO(audio_file.read()) as wav_buffer:
                samplerate, data = wavfile.read(wav_buffer)
            
            audio_file.seek(original_position)

            max_amplitude = np.max(np.abs(data))
            return max_amplitude <= max_amplitude_threshold
        except Exception as e:
            logger.error(f"Error while checking for silence: {str(e)}")
            return True

    async def transcribe_audio(self, audio_file: BinaryIO) -> Optional[str]:
        try:
            original_position = audio_file.tell()
            audio_file.seek(0)
            
            transcription = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            
            audio_file.seek(original_position)
            return str(transcription) if transcription else None
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return None

# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    agent = VoiceAgent(api_key=api_key)

    async def main():
        test_file_path = "test.wav"  # Make sure this file exists in your directory
        if not os.path.exists(test_file_path):
            print(f"Test file not found: {test_file_path}")
            return

        try:
            with open(test_file_path, "rb") as audio_file:
                result = await agent.process_audio_file(audio_file)

            if result:
                print("Transcription:", result)
            else:
                print("No transcription available.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    asyncio.run(main())