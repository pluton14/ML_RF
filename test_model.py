import os
import numpy as np
from scipy import signal
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def load_and_predict(model_path, test_dir):
    """Тестирование модели на 1-секундных сегментах"""
    model = tf.keras.models.load_model(model_path)
    class_names = sorted(os.listdir(test_dir))

    # Собираем все тестовые данные
    X_test = []
    y_test = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.bin')][:100]  # Берем по 100 на класс

        for file in files:
            file_path = os.path.join(class_dir, file)
            iq_data = np.fromfile(file_path, dtype=np.complex64)

            # Преобразование в спектрограмму (как при обучении)
            f, t, Zxx = signal.stft(
                iq_data,
                fs=2.4e6,
                nperseg=1024,
                noverlap=512,
                return_onesided=True
            )
            Sxx = np.abs(Zxx)
            Sxx_db = 20 * np.log10(Sxx + 1e-10)
            Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db))

            X_test.append(Sxx_norm[..., np.newaxis])
            y_test.append(class_idx)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Отчет классификации
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Примеры предсказаний
    for i in range(3):
        plt.figure(figsize=(10, 4))
        plt.imshow(X_test[i][..., 0], aspect='auto', cmap='viridis')
        true_name = class_names[y_test[i]]
        pred_name = class_names[y_pred_classes[i]]
        plt.title(f"True: {true_name}\nPred: {pred_name} ({np.max(y_pred[i]):.2f})")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    load_and_predict(
        model_path="drone_1sec_classifier.h5",
        test_dir="1sec_segments"
    )