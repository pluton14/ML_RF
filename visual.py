import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

file_path = r"C:\Users\Платон\.cache\kagglehub\datasets\zhaoericry\drone-rf-dataset\versions\2\Phantom4"

def process_iq_file(file_path, sample_rate=56e6, center_freq=2.4e9, max_samples=10e6):
    """Чтение и визуализация I/Q данных"""
    try:
        # Определяем количество сэмплов
        file_size = os.path.getsize(file_path)
        sample_size = 8  # complex64 = 2*float32 (4+4 байта)
        total_samples = file_size // sample_size

        # Читаем только часть данных (для производительности)
        samples_to_read = min(total_samples, int(max_samples))

        # Чтение данных
        iq_data = np.fromfile(file_path,
                              dtype=np.complex64,
                              count=samples_to_read)

        print(f"Прочитано {len(iq_data)} сэмплов")

        # Визуализация
        plt.figure(figsize=(15, 10))

        # 1. Временной график
        # plt.subplot(3, 1, 1)
        # plt.plot(np.real(iq_data[:1000]), label='I')
        # plt.plot(np.imag(iq_data[:1000]), label='Q')
        # plt.title("I/Q Samples (First 1000 points)")
        # plt.legend()

        # 2. Спектр
        plt.subplot(3, 1, 1)
        spectrum = np.fft.fftshift(np.fft.fft(iq_data))
        freq = np.linspace(-sample_rate / 2, sample_rate / 2, len(spectrum))
        plt.plot(freq / 1e6, 20 * np.log10(np.abs(spectrum)))
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Magnitude [dB]")

        # 3. Спектрограмма
        plt.subplot(3, 1, 2)
        f, t, Sxx = signal.spectrogram(
            iq_data,
            fs=sample_rate,
            nperseg=1024,
            noverlap=512,
            return_onesided=False
        )
        plt.pcolormesh(t, np.fft.fftshift(f) / 1e6,
                       10 * np.log10(np.fft.fftshift(Sxx, axes=0)),
                       shading='auto')
        plt.title("Spectrogram")
        plt.ylabel("Frequency [MHz]")
        plt.xlabel("Time [sec]")
        plt.colorbar(label="Power [dB]")

        plt.tight_layout()
        plt.show()

        return iq_data

    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return None

# Проверка существования файла с учетом юникода
if os.path.exists(file_path):
    print(f"Файл существует. Размер: {os.path.getsize(file_path)} байт")

iq_data = process_iq_file(file_path)

