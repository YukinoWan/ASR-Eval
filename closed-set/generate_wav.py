from pathlib import Path
import json
import random
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://llm-proxy.perflab.nvidia.com",
    api_key="",
)

# Voice metadata dictionary
VOICE_INFO = {
    "alloy": {"gender": "female"},
    "echo": {"gender": "male"},
    "fable": {"gender": "female"},
    "onyx": {"gender": "male"},
    "nova": {"gender": "female"},
    "shimmer": {"gender": "male"}
}

# Use target_wav_data directory
output_dir = Path(__file__).parent / "target_wav_data"
output_dir.mkdir(exist_ok=True)

# Load the input descriptions
with open(Path(__file__).parent / "whisper_input_descriptions.json", "r") as f:
    input_data = json.load(f)
    input_data["texts"] = input_data["texts"][:10]

# Initialize metadata storage
generation_metadata = []

# Process each text entry
for text_entry in input_data["texts"]:
    # Select random voice
    selected_voice = random.choice(list(VOICE_INFO.keys()))
    
    # Create unique filename using the ID
    audio_filename = f"speech_{text_entry['id']}_{selected_voice}.mp3"
    speech_file_path = output_dir / audio_filename

    try:
        # Generate speech
        response = client.audio.speech.create(
            model="tts",
            voice=selected_voice,
            input=text_entry["content"]
        )

        # Save the audio file
        response.stream_to_file(str(speech_file_path))

        # Store metadata
        generation_metadata.append({
            "id": text_entry["id"],
            "voice": selected_voice,
            "gender": VOICE_INFO[selected_voice]["gender"],
            "text": text_entry["content"],
            "audio_file": audio_filename
        })

        print(f"Generated audio for ID {text_entry['id']} using voice {selected_voice}")

    except Exception as e:
        print(f"Error generating audio for ID {text_entry['id']}: {str(e)}")

# Save metadata to JSON file
metadata_file = output_dir / "generation_metadata.json"
with open(metadata_file, "w") as f:
    json.dump({"generations": generation_metadata}, f, indent=2)

print(f"\nGeneration complete. Generated {len(generation_metadata)} audio files.")
print(f"Metadata saved to {metadata_file}")
