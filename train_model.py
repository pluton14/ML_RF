import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt


# 1. Настройки памяти
def configure_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.config.set_soft_device_placement(True)


configure_memory()

# 2. Параметры
DATA_DIR = "1sec_segments"
BATCH_SIZE = 128
EPOCHS = 10
SAMPLE_RATE = 2.4e6
NPERSEG = 256  # Уменьшенный размер окна
NOVERLAP = 128  # Уменьшенное перекрытие


# 3. Генератор данных
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=16):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        batch_spectrograms = []
        for path in batch_paths:
            try:
                iq_data = np.fromfile(path, dtype=np.complex64)
                _, _, Zxx = signal.stft(
                    iq_data,
                    fs=SAMPLE_RATE,
                    nperseg=NPERSEG,
                    noverlap=NOVERLAP,
                    return_onesided=True
                )
                Sxx = np.abs(Zxx)
                Sxx = 20 * np.log10(Sxx + 1e-10)
                Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
                batch_spectrograms.append(Sxx[..., np.newaxis])
            except Exception as e:
                print(f"Ошибка обработки {path}: {str(e)}")
                continue

        return np.array(batch_spectrograms), np.array(batch_labels)

    def on_epoch_end(self):
        indices = np.arange(len(self.file_paths))
        np.random.shuffle(indices)
        self.file_paths = [self.file_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]


# 4. Упрощенная модель (всего ~15K параметров)
def create_tiny_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Миниатюрная архитектура
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# 5. Основной процесс
def main():
    # Загрузка данных
    class_names = sorted([d for d in os.listdir(DATA_DIR)
                          if os.path.isdir(os.path.join(DATA_DIR, d))])
    file_paths = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        files = [os.path.join(class_dir, f)
                 for f in os.listdir(class_dir)
                 if f.endswith('.bin')]
        file_paths.extend(files)
        labels.extend([class_idx] * len(files))

    # Разделение данных
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Определение input_shape
    try:
        sample_iq = np.fromfile(train_paths[0], dtype=np.complex64)
        _, _, Zxx = signal.stft(
            sample_iq,
            fs=SAMPLE_RATE,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            return_onesided=True
        )
        input_shape = (Zxx.shape[0], Zxx.shape[1], 1)
        print(f"Input shape: {input_shape}")
    except Exception as e:
        print(f"Ошибка определения формы: {str(e)}")
        input_shape = (129, 234, 1)  # Значение по умолчанию

    # Создание генераторов
    train_gen = DataGenerator(train_paths, train_labels, BATCH_SIZE)
    val_gen = DataGenerator(val_paths, val_labels, BATCH_SIZE)

    # Создание модели
    model = create_tiny_model(input_shape, len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Вывод информации о модели
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.weights.h5',  # Исправленное имя файла
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
    ]

    # Обучение
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Сохранение и визуализация
    model.save_weights('final_model.weights.h5')  # Исправленное имя файла

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


if __name__ == "__main__":
    main()