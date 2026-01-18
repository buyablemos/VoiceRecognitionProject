import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os 

# Funkcje do wizualizacji wyników trenowania modelu

def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Wykres Dokładności
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Trening Accuracy')
    plt.plot(epochs_range, val_acc, label='Walidacja Accuracy')
    plt.legend(loc='lower right')
    plt.title('Dokładność (Accuracy)')
    plt.grid(True)

    # Wykres Straty
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Trening Loss')
    plt.plot(epochs_range, val_loss, label='Walidacja Loss')
    plt.legend(loc='upper right')
    plt.title('Funkcja Straty (Loss)')
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig("learning_curves.png")
    plt.close()
    print(f"Zapisano wykres uczenia: learning_curves.png")

def plot_confusion_matrix(model, X_test, y_test, class_names):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Przewidziana etykieta')
    plt.ylabel('Prawdziwa etykieta')
    plt.title('Macierz Pomyłek')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig("confusion_matrix.png")
    plt.close()
    print(f"Zapisano macierz pomyłek: confusion_matrix.png")
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open("raport_klasyfikacji.txt", "w") as f:
        f.write(report)
    print("Zapisano raport tekstowy.")

def visualize_sample_input(X, y, class_names, index=0):
    plt.figure(figsize=(4, 4))
    spec_data = X[index].reshape(X[index].shape[0], X[index].shape[1])
    
    plt.imshow(spec_data, aspect='auto', origin='lower', cmap='inferno')
    plt.title(f"Próbka: {class_names[y[index]]}")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    plt.savefig("sample_spectrogram.png")
    plt.close()
    print(f"Zapisano przykładowy spektrogram: sample_spectrogram.png")