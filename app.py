import streamlit as st
import os
import json
import shutil
import re
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import boto3  # For Amazon Bedrock integration

# Load environment variables from .env file
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Podcast Generator", layout="wide")
st.title("🎙️ Podcast Generator")

# Retrieve ElevenLabs API key from environment
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Define URLs for each speaker's voice
elevenlabs_voice_urls = {
    "Brian": "https://api.elevenlabs.io/v1/text-to-speech/wAbpJY3NBvWoMLWJrKAD",  # Brian's Voice ID URL
    "Marina": "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL"  # Marina's Voice ID URL
}

elevenlabs_headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": elevenlabs_api_key
}

# Amazon Bedrock client setup
bedrock_client = boto3.client('bedrock-runtime')

# ElevenLabs TTS function
def synthesize_speech_elevenlabs(text, speaker, index):
    # Select the URL based on the speaker
    elevenlabs_url = elevenlabs_voice_urls.get(speaker)
    if not elevenlabs_url:
        raise ValueError(f"Voice URL for speaker '{speaker}' not found.")
    
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(elevenlabs_url, json=data, headers=elevenlabs_headers)
    filename = f"audio-files/{index}_{speaker}.mp3"
    with open(filename, "wb") as out:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                out.write(chunk)

# Function to synthesize speech based on the speaker
def synthesize_speech(text, speaker, index):
    synthesize_speech_elevenlabs(text, speaker, index)

# Function to sort filenames naturally
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

# Function to merge audio files
def merge_audios(audio_folder, output_file):
    combined = AudioSegment.empty()
    audio_files = sorted(
        [f for f in os.listdir(audio_folder) if f.endswith(".mp3") or f.endswith(".wav")],
        key=natural_sort_key
    )
    for filename in audio_files:
        audio_path = os.path.join(audio_folder, filename)
        audio = AudioSegment.from_file(audio_path)
        combined += audio
    combined.export(output_file, format="mp3")

# Function to generate conversation using Amazon Bedrock
def generate_conversation(article):
    system_prompt = """
    You are an experienced podcast host. Based on text like an article, you can create an engaging conversation between two people, Brian and Marina.
    Make the conversation long, natural, and engaging, with excitement and emotion. 
    Brian is writing the articles, and Marina is the second speaker asking insightful questions.
    """
    
    # Combine the prompt with the article
    prompt_text = system_prompt + "\n\n" + article
    model_id = "mistral.mistral-7b-instruct-v0:2"  # Example model ID; replace with appropriate Bedrock model

    # Send request to Amazon Bedrock without the `textGenerationConfig` key
    response = bedrock_client.invoke_model(
        body=json.dumps({
            "inputText": prompt_text
        }),
        modelId=model_id,
        accept='application/json',
        contentType='application/json'
    )

    # Parse response and convert it to JSON format
    json_response = json.loads(response['body'].read())
    conversation_text = json_response['results'][0]['outputText']
    
    # Simulate conversation format based on returned text, detecting speaker prefixes
    conversation = []
    current_speaker = "Brian"  # Start with Brian as the first speaker
    lines = conversation_text.splitlines()
    
    for line in lines:
        # Use regex to detect if a line starts with a speaker's name
        match = re.match(r"^(Brian|Marina):\s*(.*)", line)
        if match:
            speaker, text = match.groups()
            conversation.append({"speaker": speaker, "text": text})
            current_speaker = speaker  # Update current speaker
        else:
            # If no speaker is detected, alternate between speakers
            text = line.strip()
            if text:  # Avoid appending empty text lines
                conversation.append({"speaker": current_speaker, "text": text})
                # Alternate the speaker for the next line
                current_speaker = "Marina" if current_speaker == "Brian" else "Brian"
    
    return conversation

# Function to generate the podcast audio from conversation data
def generate_audio(conversation):
    if os.path.exists('audio-files'):
        shutil.rmtree('audio-files')
    os.makedirs('audio-files', exist_ok=True)
    
    for index, part in enumerate(conversation):
        speaker = part['speaker']
        text = part['text']
        synthesize_speech(text, speaker, index)
    
    output_file = "podcast.mp3"
    merge_audios("audio-files", output_file)
    return output_file

# Streamlit inputs and outputs
article = st.text_area("Article Content", "Paste the article text here", height=300)
if st.button("Generate Podcast"):
    if not article:
        st.error("Please enter article content to generate a podcast.")
    else:
        with st.spinner("Generating conversation..."):
            conversation = generate_conversation(article)
        
        st.success("Conversation generated successfully!")
        st.json(conversation)
        
        # Generate audio files
        with st.spinner("Synthesizing audio..."):
            podcast_file = generate_audio(conversation)
        
        st.success("Audio synthesis complete!")
        st.audio(podcast_file, format="audio/mp3")
        st.download_button("Download Podcast", data=open(podcast_file, "rb"), file_name="podcast.mp3", mime="audio/mp3")
