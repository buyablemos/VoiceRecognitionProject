import os
import librosa
import numpy as np
import gc
from sklearn.model_selection import train_test_split

SAMPLES_TO_LOAD = 16000 # 1 sekunda przy 16kHz

def augment_audio(y, sr):
    choice = np.random.choice(['noise', 'shift', 'pitch', 'speed'])
    
    if choice == 'noise':
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
    elif choice == 'shift':
        shift_range = int(np.random.uniform(-0.1, 0.1) * sr)
        y = np.roll(y, shift_range)
    elif choice == 'pitch':
        steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif choice == 'speed':
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
        
    # Przycinanie / dopełnianie do stałej długości
    if len(y) < SAMPLES_TO_LOAD:
        y = np.pad(y, (0, SAMPLES_TO_LOAD - len(y)))
    else:
        y = y[:SAMPLES_TO_LOAD]
    return y

def prepare_single_file(file_path, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=1.0)
    except Exception as e:
        print(f"Błąd pliku {file_path}: {e}")
        return np.zeros((128, 32, 1), dtype=np.float16)
    
    if len(y) == 0: return np.zeros((128, 32, 1), dtype=np.float16)

    # Normalizacja audio
    y = y / (np.max(np.abs(y)) + 1e-6)
    
    if len(y) < SAMPLES_TO_LOAD:
        y = np.pad(y, (0, SAMPLES_TO_LOAD - len(y)))
    else:
        y = y[:SAMPLES_TO_LOAD]

    if augment:
        y = augment_audio(y, sr)

    # Spektrogram Mel
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalizacja 0-1
    min_val, max_val = np.min(mel_spec_db), np.max(mel_spec_db)
    if max_val - min_val > 0:
        mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
    else:
        mel_spec_db = np.zeros_like(mel_spec_db)

    return mel_spec_db.reshape(128, mel_spec_db.shape[1], 1)

def generate_data_array(paths, labels, augment_factor=0):
    X_list = []
    y_list = []
    
    total = len(paths)
    print(f"Generowanie danych (Augmentacja: +{augment_factor} kopii)")
    
    for i, (path, label) in enumerate(zip(paths, labels)):
        #Oryginał
        X_list.append(prepare_single_file(path, augment=False))
        y_list.append(label)
        
        #Augmentacja (tylko jeśli augment_factor > 0)
        for _ in range(augment_factor):
            X_list.append(prepare_single_file(path, augment=True))
            y_list.append(label)
            
    return np.array(X_list), np.array(y_list)

def load_dataset(data_path, test_size=0.2, augment_factor=15):
   
    print(f"ROZPOCZĘTO ŁADOWANIE DANYCH Z: {data_path}")
    
    #Zbieranie ścieżek
    paths = []
    labels = []
    label_map = {}
    
    # Struktura: dataset/Osoba/Komenda/plik.wav
    speakers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for speaker in speakers:
        s_path = os.path.join(data_path, speaker)
        cmds = [d for d in os.listdir(s_path) if os.path.isdir(os.path.join(s_path, d))]
        
        for cmd in cmds:
            c_path = os.path.join(s_path, cmd)
            files = [f for f in os.listdir(c_path) if f.endswith('.wav')]
            
            # Etykieta to np. "Dawid_start"
            class_name = f"{speaker}_{cmd}"
            if class_name not in label_map:
                label_map[class_name] = len(label_map)
            
            idx = label_map[class_name]
            
            for f in files:
                paths.append(os.path.join(c_path, f))
                labels.append(idx)
    
    print(f"Znaleziono {len(paths)} plików. Klasy: {len(label_map)}")
    
    # Podział na zbiory (na poziomie ścieżek)
    X_train_paths, X_test_paths, y_train_labels, y_test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    del paths, labels
    gc.collect()
    
    # Generowanie ciężkich danych
    print("Przetwarzanie zbioru TESTOWEGO (tylko oryginały)")
    X_test, y_test = generate_data_array(X_test_paths, y_test_labels, augment_factor=0)
    
    print("Przetwarzanie zbioru TRENINGOWEGO (z augmentacją)")
    X_train, y_train = generate_data_array(X_train_paths, y_train_labels, augment_factor=augment_factor)
    
    del X_train_paths, X_test_paths
    gc.collect()
    
    # Sortowanie nazw klas wg indeksów, żeby np. 0=Dawid_start, 1=Dawid_stop
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    print("ZAKOŃCZONO ŁADOWANIE")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, class_names