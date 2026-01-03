import os
import librosa
import numpy as np
import tensorflow as tf
import sounddevice as sd

# --- KONFIGURACJA ---
MODEL_PATH = "model_komend.h5"
LABELS_PATH = "labels.npy"
FS = 16000
DURATION = 1.0 
THRESHOLD = 0.02 # Jeśli nic nie wykrywa, zmniejsz do 0.01
CHUNK_SIZE = 1600 # Paczki po 0.1s


if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("Błąd: Brakuje plików modelu lub etykiet!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
LABELS = np.load(LABELS_PATH)

def preprocess_audio(audio_data):
    y = audio_data.flatten()
    target_len = int(FS * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    mel_spec = librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalizacja (taka sama jak przy uczeniu)
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-6)
    return mel_spec_db.reshape(1, 128, mel_spec_db.shape[1], 1)

def run_recognition():
    print(f"Wczytano klasy: {LABELS}")
    print("\n[NASŁUCHIWANIE] Czekam na głos... (Naciśnij Ctrl+C, aby wyłączyć)")

    # Otwieramy strumień
    with sd.InputStream(samplerate=FS, channels=1) as stream:
        while True:
            audio_chunk, overflow = stream.read(CHUNK_SIZE)
            
            # Oblicz głośność (RMS)
            rms = np.sqrt(np.mean(audio_chunk**2))
            
            if rms > THRESHOLD:
                print(f" -> Wykryto dźwięk (RMS: {rms:.4f})")
                
                # Czytamy resztę sekundy ze strumienia, żeby mieć pełną komendę
                remaining_samples = int(FS * DURATION) - CHUNK_SIZE
                extra_audio, overflow = stream.read(remaining_samples)
                
                # Łączymy początek z resztą
                full_audio = np.concatenate([audio_chunk, extra_audio])
                
                # Analiza
                input_data = preprocess_audio(full_audio)
                prediction = model.predict(input_data, verbose=0)
                
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                
                if confidence > 0.7:
                    print(f"   >>> ROZPOZNANO: {LABELS[class_idx].upper()} ({confidence*100:.0f}%)")
                else:
                    print("   [?] Dźwięk zbyt niepewny.")
                
                print("...gotowy na kolejną komendę.")

if __name__ == "__main__":
    try:
        run_recognition()
    except KeyboardInterrupt:
        print("\nZatrzymano program.")