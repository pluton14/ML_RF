import os
import numpy as np
from tqdm import tqdm
import shutil

# Параметры
SAMPLE_RATE = 2.4e6  # 2.4 MHz
SEGMENT_LENGTH = int(1 * SAMPLE_RATE)  # Ровно 1 секунда
MIN_POWER = -60  # Минимальная мощность сигнала в dB
OUTPUT_DIR = "1sec_segments"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def calculate_power(iq_segment):
    """Вычисление мощности сигнала в dB"""
    power = np.mean(np.abs(iq_segment) ** 2)
    return 10 * np.log10(power + 1e-10)


def split_large_file(input_path, output_dir, class_name):
    """Разбиение большого файла на 1-секундные сегменты"""
    file_size = os.path.getsize(input_path)
    sample_size = 8  # complex64 = 8 байт
    total_samples = file_size // sample_size
    num_segments = total_samples // SEGMENT_LENGTH

    print(f"\nОбработка {class_name}...")
    print(f"Всего сегментов: {num_segments}")

    valid_segments = 0
    with open(input_path, 'rb') as f:
        for i in tqdm(range(num_segments)):
            # Чтение ровно 1 секунды (2.4M сэмплов)
            iq_data = np.fromfile(f, dtype=np.complex64, count=SEGMENT_LENGTH)

            # Проверка мощности (отсекаем тишину/шум)
            if calculate_power(iq_data) > MIN_POWER:
                output_path = os.path.join(output_dir, class_name, f"{class_name}_{i:04d}.bin")
                iq_data.tofile(output_path)
                valid_segments += 1

    print(f"Сохранено сегментов с сигналом: {valid_segments}/{num_segments}")
    return valid_segments


def main():
    ensure_dir(OUTPUT_DIR)

    # Исходные файлы (ваши 3 больших файла)
    DRONE_FILES = {
        "FIMI": r"C:\Users\Платон\.cache\kagglehub\datasets\zhaoericry\drone-rf-dataset\versions\2\FIMI",
        "Phantom4": r"C:\Users\Платон\.cache\kagglehub\datasets\zhaoericry\drone-rf-dataset\versions\2\Phantom4",
        "Parrot": r"C:\Users\Платон\.cache\kagglehub\datasets\zhaoericry\drone-rf-dataset\versions\2\Parrot"
    }

    # Создаем папки для каждого класса
    for class_name in DRONE_FILES:
        ensure_dir(os.path.join(OUTPUT_DIR, class_name))

    # Обработка всех файлов
    total = 0
    for class_name, file_path in DRONE_FILES.items():
        if os.path.exists(file_path):
            cnt = split_large_file(file_path, OUTPUT_DIR, class_name)
            total += cnt
        else:
            print(f"Файл не найден: {file_path}")

    print(f"\nВсего сохранено 1-секундных сегментов: {total}")
    print(f"Путь к данным: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()