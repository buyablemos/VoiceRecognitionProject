import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- KONFIGURACJA ---
DATA_PATH = "dataset"
IMG_SIZE = (128, 32)  # Rozmiar spektrogramu (częstotliwość x czas)
SAMPLES_TO_LOAD = 16000 # 1 sekunda przy 16kHz

def augment_audio(y, sr):
    choice = np.random.choice(['noise', 'shift', 'pitch', 'speed', 'none'])
    
    if choice == 'noise':
        # Dodanie białego szumu (symuluje szum mikrofonu)
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
        
    elif choice == 'shift':
        # Przesunięcie w czasie (ważne, gdy nie trafisz idealnie w sekundę)
        shift_range = int(np.random.uniform(-0.2, 0.2) * sr)
        y = np.roll(y, shift_range)
        
    elif choice == 'pitch':
        # Zmiana wysokości tonu (symuluje inną intonację)
        steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        
    elif choice == 'speed':
        # Przyspieszenie lub spowolnienie (symuluje tempo mowy)
        rate = np.random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate=rate)
        
    if len(y) < SAMPLES_TO_LOAD:
        y = np.pad(y, (0, SAMPLES_TO_LOAD - len(y))) # Dopełniamy ciszą, jeśli po przyspieszeniu jest za krótki
    else:
        y = y[:SAMPLES_TO_LOAD]

    return y

def prepare_audio(file_path, augment=False):
    
    y, sr = librosa.load(file_path, sr=16000, duration=1.0)
    
    if len(y) > 0:
        y = y / (np.max(np.abs(y)) + 1e-6)
    
    # AUGMENTACJA
    if augment:
        y = augment_audio(y, sr)

    # Zamiana na Spektrogram Mel
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalizacja do zakresu 0-1
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
    
    # Zmiana wymiarów pod CNN
    return mel_spec_db.reshape(128, mel_spec_db.shape[1], 1)


def load_dataset():
    X, y = [], []
    label_map = {}
    
    # Przeszukujemy folder: dataset/osoba/komenda/plik.wav
    # Tworzymy etykietę łączoną: "osoba_komenda"
    speakers = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    
    for speaker in speakers:
        speaker_path = os.path.join(DATA_PATH, speaker)

        commands = [d for d in os.listdir(speaker_path) if os.path.isdir(os.path.join(speaker_path, d))]

        for cmd in commands:
            cmd_path = os.path.join(speaker_path, cmd)
            files = [f for f in os.listdir(cmd_path) if f.endswith('.wav')]
            
            label = f"{speaker}_{cmd}"
            if label not in label_map:
                label_map[label] = len(label_map)
            
            print(f"Przetwarzanie: {label}...")
            
            for f in files:
                full_path = os.path.join(cmd_path, f)
                # 1. Dodaj oryginał
                X.append(prepare_audio(full_path, augment=False))
                y.append(label_map[label])
                
                # 2. Dodaj różne wersje zaszumione/przesunięte
                for _ in range(15):
                    X.append(prepare_audio(full_path, augment=True))
                    y.append(label_map[label])
                
    return np.array(X), np.array(y), label_map


X, y, label_map = load_dataset()
num_classes = len(label_map)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Budowa modelu CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, X.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') # Klasyfikacja na N klas (osoba_komenda)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie
print("\nRozpoczynanie uczenia...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Zapis modelu
model.save("model_komend.h5")
print("\nModel zapisany jako model_komend.h5")
print("Rozpoznawane klasy:", label_map)

# label_map to słownik { 'osoba_komenda': indeks }, który jest potrzebny do mapowania wyników predykcji z powrotem na etykiety
labels_to_save = sorted(label_map.items(), key=lambda x: x[1])
labels_list = [l[0] for l in labels_to_save]
np.save("labels.npy", labels_list)

print(f"Lista etykiet ({len(labels_list)} klas) została zapisana w labels.npy")