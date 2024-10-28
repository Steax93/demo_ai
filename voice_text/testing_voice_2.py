import json
import logging
import os
from typing import Optional, Dict, Any

from openai import OpenAI
import numpy as np
from scipy.io import wavfile
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info('VoiceAgent initialized')

    def run(self, file_path: str) -> Dict[str, Any]:
        logger.info(f"Starting VoiceAgent run for file: {file_path}")
        
        try:
            processed_audio = self.process_audio_file(file_path)
            if processed_audio is None:
                return {"status": "error", "message": "Audio processing failed"}

            transcription = self.transcribe_audio(processed_audio)
            if transcription is None:
                return {"status": "error", "message": "Transcription failed"}

            result = {
                "status": "success",
                "transcription": transcription
            }
            logger.info(f"VoiceAgent run completed. Status: {result['status']}")
            return result

        except Exception as e:
            logger.error(f"Error in VoiceAgent run: {str(e)}")
            return {"status": "error", "message": str(e)}

    def process_audio_file(self, file_path: str) -> Optional[str]:
        logger.info(f'Processing audio file: {file_path}')
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        file_format = self._check_audio_format(file_path)
        if not file_format:
            logger.error(f"Invalid audio format for file: {file_path}")
            return None

        if file_format == 'mp3':
            try:
                file_path = self._convert_to_wav(file_path)
                logger.info(f"Converted MP3 to WAV: {file_path}")
            except Exception as e:
                logger.error(f"Error converting MP3 to WAV: {str(e)}")
                return None

        if self._is_silence(file_path):
            logger.info("File contains only silence")
            return None

        return file_path

    def _check_audio_format(self, file_path: str) -> Optional[str]:
        _, extension = os.path.splitext(file_path)
        if extension.lower() in ['.wav', '.mp3']:
            return extension[1:].lower()
        return None

    def _convert_to_wav(self, file_path: str) -> str:
        wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        try:
            result = subprocess.run(['ffmpeg', '-i', file_path, wav_path], check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg conversion output: {result.stdout}")
            return wav_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg and make sure it's in your system PATH.")
            raise

    def _is_silence(self, file_path: str, max_amplitude_threshold: int = 3000) -> bool:
        try:
            samplerate, data = wavfile.read(file_path)
            max_amplitude = np.max(np.abs(data))
            is_silent = max_amplitude <= max_amplitude_threshold
            logger.info(f"Silence check result: {'Silent' if is_silent else 'Not silent'}")
            return is_silent
        except Exception as e:
            logger.error(f"Error while checking for silence: {str(e)}")
            return True

    def transcribe_audio(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'rb') as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            logger.info(f"Transcription completed: {transcription[:50]}...")
            return transcription
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return None

# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    agent = VoiceAgent(api_key)

    result = agent.run("test.mp3")
    print("Result:", json.dumps(result, ensure_ascii=False, indent=2))