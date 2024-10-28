
import numpy as np
import os 
import openai
import random
from scipy.io.wavfile import write
import sounddevice as sd
import pyttsx3

# Set your OpenAI API key here
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global variable to store audio data
audio = []

# Adjectives to generate random names for voices
adjectives = ["beautiful", "sad", "mystical", "serene", "whispering", "gentle", "melancholic"]
nouns = ["sea", "love", "dreams", "song", "rain", "sunrise", "silence", "echo"]
#initializing pytts for text to speech output
engine = pyttsx3.init()
engine.setProperty('rate', 130)

 
def generate_random_name():
    # to generate random unique names for the audio voice recordings
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    return f"{adjective} {noun}"

def new_record_audio():
    # to record audio as wav file
    print("Recording... Press 's' to stop.")
    fs = 44100
    seconds = 6
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    audio_name = generate_random_name()
    write(f'./{audio_name}.wav', fs, myrecording)  # Save as WAV file 
    print("Recording stopped.")
    return f'./{audio_name}.wav'
    
def transcribe_audio(audio_path):
    print ("entered transcribe", "./"+audio_path)
    audio_file= open(audio_path, "rb")
    print(audio_file)
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
    return transcript['text']



def speech_to_text(response):
    # to generate the final output voice from text
    engine.say(response)
    engine.runAndWait()




def main():
    while True:
        print("Press 's' to stop recording and transcribe the audio.")
        # Start recording live voice input
        recorded_audio_path = new_record_audio()
        print("Recording stopped. Transcribing audio...")
        # Save the recorded audio as a WAV file
        print("Recorded audio saved to:", recorded_audio_path)
        print("----end---")
        # Transcribe the audio
        transcript = transcribe_audio(recorded_audio_path)
        # Create a list of messages with the user's input
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcript}
        ]
        print("Transcript:")
        print(transcript)
        # Make the API call for gpt AI
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response = completion.choices[0].message["content"]
        # Print the assistant's response
        print("Assistant:", response)
        
        # Convert output to voice
        speech_to_text(response)

        # Ask whether to continue or stop
        user_choice = input("Continue? (y/n): ")
        if user_choice.lower() != "y":
            print("Glad to help bye!")
            break  # Exit the loop if the user doesn't want to continue

if __name__ == "__main__":
        main()