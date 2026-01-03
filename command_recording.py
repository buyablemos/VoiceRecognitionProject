import sounddevice as sd
from scipy.io import wavfile
import os
import time

# --- KONFIGURACJA ---
COMMANDS = [
    "start", "stop", "lewo", "prawo", "góra", "dół", 
    "otwórz", "zamknij", "jasność", "głośność", "wyłącz", "aktywuj"
]
SPEAKERS = ["Dawid", "Radek"]
DURATION = 1  # Czas trwania nagrania w sekundach
FS = 16000    # Częstotliwość próbkowania (16kHz jest standardem dla mowy)
SAMPLES_PER_COMMAND = 10  # Ile razy powtórzyć każdą komendę na sesję

def record_audio(speaker_name):
    base_path = f"dataset/{speaker_name}"
    
    print(f"\n--- SESJA NAGRANIOWA DLA: {speaker_name} ---")
    
    for command in COMMANDS:
        folder_path = os.path.join(base_path, command)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"\nPrzygotuj się do mówienia komendy: [{command.upper()}]")
        time.sleep(0.5)
        
        for i in range(SAMPLES_PER_COMMAND):
            # Nazwa pliku z timestampem, aby nie nadpisywać przy kolejnych sesjach
            file_name = f"{command}_{int(time.time())}_{i}.wav"
            file_path = os.path.join(folder_path, file_name)
            
            print(f"  -> Nagrywanie próbki {i+1}/{SAMPLES_PER_COMMAND}...")
            
            # Nagrywanie
            recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
            sd.wait()  # Czekaj na zakończenie nagrywania
            
            # Zapis do pliku
            wavfile.write(file_path, FS, recording)
            
        print(f"Zakończono serię dla: {command}")

if __name__ == "__main__":
    print("Dostępni mówcy:", SPEAKERS)
    current_speaker = input("Kto nagrywa? (wpisz nazwę): ")
    
    if current_speaker in SPEAKERS:
        record_audio(current_speaker)
        print("\nNagrywanie zakończone pomyślnie!")
    else:
        print("Błąd: Nie ma takiego mówcy w konfiguracji.")