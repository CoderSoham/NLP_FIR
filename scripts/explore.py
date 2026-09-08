import os
import whisper
import librosa
import matplotlib.pyplot as plt
import numpy as np
import spacy
import librosa.display
from pydub import AudioSegment

# Load models
print("Loading models...")
whisper_model = whisper.load_model("large")
nlp = spacy.load("en_core_web_sm")
print("Models loaded.")

# Get file from user
input_path = input("Enter the path to your .mp3 audio file: ").strip()
if not input_path.lower().endswith('.mp3') or not os.path.exists(input_path):
    raise Exception("Invalid or non-existent MP3 file!")

# Convert to WAV and trim to 2 minutes
print("Converting to WAV...")
audio = AudioSegment.from_file(input_path, format="mp3")[:120000]
temp_wav_path = "temp.wav"
audio.export(temp_wav_path, format="wav")

# Transcribe
print("Transcribing...")
result = whisper_model.transcribe(temp_wav_path)
transcription = result["text"].strip().lower()
print(f"\n--- Transcription ---\n{transcription}\n")

# NLP - NER
doc = nlp(transcription)
entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

print("--- Named Entities ---")
for ent in entities:
    print(f"{ent['text']} ({ent['label']})")

# Audio Features
print("Extracting features...")
y, sr = librosa.load(temp_wav_path)

# Make sure plots directory exists
os.makedirs("plots", exist_ok=True)

# 1. Pitch (median per frame, robust filtering)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch_values = []
for i in range(pitches.shape[1]):
    index = magnitudes[:, i].argmax()
    pitch = pitches[index, i]
    if pitch > 0:
        pitch_values.append(pitch)

plt.figure()
plt.plot(pitch_values)
plt.title("Pitch Contour")
plt.xlabel("Frame")
plt.ylabel("Pitch (Hz)")
plt.tight_layout()
plt.savefig("plots/pitch.png")
plt.close()

# 2. Waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig("plots/waveform.png")
plt.close()

# 3. MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.savefig("plots/mfcc.png")
plt.close()

# 4. NER Distribution
if entities:
    labels = [e['label'] for e in entities]
    plt.figure()
    plt.hist(labels, bins=len(set(labels)))
    plt.title("NER Entity Distribution")
    plt.xlabel("Entity Type")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/entities.png")
    plt.close()

print("\n✅ Done. Transcription, entities, and graphs saved in the 'plots' folder.")
