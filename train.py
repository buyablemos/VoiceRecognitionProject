import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from visu import plot_learning_curves, plot_confusion_matrix, visualize_sample_input
from train_prepare import load_dataset


DATA_PATH = "dataset"

X_train, X_test, y_train, y_test, class_names = load_dataset(DATA_PATH)
num_classes = len(class_names)

# Budowa modelu CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, X_train.shape[2], 1)),
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

print("\nRozpoczynanie uczenia")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Zapis modelu
model.save("model_komend.h5")
print("\nModel zapisany jako model_komend.h5")
print("Rozpoznawane klasy:", class_names)

np.save("labels.npy", class_names)
print("Etykiety zapisane do pliku labels.npy!")

print(f"Lista etykiet ({len(class_names)} klas) została zapisana w labels.npy")

# Wyświetlenie przykładowego spektrogramu 
print("Przykładowy spektrogram wejściowy:")
visualize_sample_input(X_test, y_test, class_names, index=0)

# Wykresy procesu uczenia
plot_learning_curves(history)

# Macierz pomyłek i raport
plot_confusion_matrix(model, X_test, y_test, class_names)