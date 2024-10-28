from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
from typing import List
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
# Initialize FastAPI app
app = FastAPI()


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Function to transcribe audio using OpenAI's API
def transcribe_audio(file_path: str):
    print("Transcribing...")
    print(f"Attempting to transcribe file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    if os.path.isfile(file_path):
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcription.text
    else:
        return None

# POST endpoint for handling audio file uploads and transcription
@app.post("/transcribe/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    results = []

    for file in files:
        # Create a temporary file with the correct extension
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            # Write the user's uploaded file to the temporary file
            content = await file.read()
            temp.write(content)
            temp.flush()

            # Close the file before passing to OpenAI
            temp.close()

            try:
                # Transcribe the temporary file using the transcribe_audio function
                transcript = transcribe_audio(temp.name)

                if transcript is None:
                    raise HTTPException(status_code=500, detail="Error transcribing audio file")

                # Store the result for this file
                results.append({
                    'filename': file.filename,
                    'transcript': transcript,
                })
            finally:
                # Clean up: remove the temporary file
                os.unlink(temp.name)

    return JSONResponse(content={'results': results})

# GET endpoint to redirect to the documentation
@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
