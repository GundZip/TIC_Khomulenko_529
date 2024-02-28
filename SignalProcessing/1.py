import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.fft import fft, fftshift, fftfreq

# Функція для генерування випадкового сигналу
def generate_random_signal(mean, std_dev, num_elements):
    return np.random.normal(mean, std_dev, num_elements)

# Функція для побудови та збереження сигналу
def plot_and_save_signal(time_values, signal_values, title, x_label, y_label, save_path=None):
    plt.figure(figsize=(21 / 2.54, 14 / 2.54))
    plt.plot(time_values, signal_values, linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()

# Створення каталогу для збереження графіків
output_directory = './figures/'
os.makedirs(output_directory, exist_ok=True)

# Параметри сигналу
n = 500  # Довжина сигналу (відліки)
Fs = 1000  # Частота дискретизації (Гц)
F_max = 31  # Максимальна частота сигналу (Гц)

# Генерація випадкового сигналу
random_signal = generate_random_signal(0, 10, n)

# Значення часу
time_values = np.arange(n) / Fs

# Параметри фільтру
F_filter = 38  # Критична частота фільтра (Гц)
w = F_filter / (Fs / 2)
filter_params = signal.butter(3, w, 'low', output='sos')

# Кроки дискретизації
discretization_steps = [2, 4, 8, 16]

# Фільтрація та відновлення аналогового сигналу з дискретного
restored_signals = []
variances = []
snr_ratios = []
discretized_signals = []

for Dt in discretization_steps:
    discrete_signal = np.zeros(n)
    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = random_signal[i * Dt]

    # Фільтрація дискретного сигналу
    restored_signal = signal.sosfiltfilt(filter_params, discrete_signal)
    restored_signals.append(restored_signal)

    # Розрахунок різниці між відновленим і початковим сигналом
    E1 = restored_signal - random_signal

    # Розрахунок дисперсій
    variance_original = np.var(random_signal)
    variance_difference = np.var(E1)
    variances.append(variance_difference)

    # Розрахунок співвідношення сигнал-шум
    snr_ratio = variance_original / variance_difference
    snr_ratios.append(snr_ratio)

    # Збереження дискретизованого сигналу
    discretized_signals.append(discrete_signal)

# Побудова графіків дискретизованих сигналів
fig, axes = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))

for i, ax in enumerate(axes.flat):
    ax.plot(time_values, discretized_signals[i], linewidth=1)
    ax.set_title(f'Дискретизований сигнал (Dt={discretization_steps[i]})')
    ax.set_xlabel('Час (с)')
    ax.set_ylabel('Значення сигналу')
    ax.grid(True)

fig.suptitle('Дискретизовані сигнали', fontsize=14)
fig.supxlabel('Час (с)', fontsize=14)
fig.supylabel('Значення сигналу', fontsize=14)
plt.tight_layout()
plt.savefig(output_directory + 'discretized_signals.png', dpi=600)
plt.show()

# Розрахунок спектрів дискретизованих сигналів
discrete_spectrums = []

for discrete_signal in discretized_signals:
    spectrum = fft(discrete_signal)
    shifted_spectrum = fftshift(spectrum)
    frequencies = fftshift(fftfreq(len(discrete_signal), 1/Fs))
    discrete_spectrums.append(shifted_spectrum)

# Відображення спектрів дискретизованих сигналів
fig, axes = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

for i, ax in enumerate(axes.flat):
    ax.plot(frequencies, abs(discrete_spectrums[i]), linewidth=1)
    ax.set_title(f'Дискретний спектр (Dt={discretization_steps[i]})')
    ax.set_xlabel('Частота (Гц)')
    ax.set_ylabel('Амплітуда')
    ax.grid(True)

fig.suptitle('Дискретні спектри дискретизованих сигналів', fontsize=14)
fig.supxlabel('Частота (Гц)', fontsize=14)
fig.supylabel('Амплітуда', fontsize=14)
plt.tight_layout()
plt.savefig(output_directory + 'discrete_spectra.png', dpi=600)
plt.show()

# Відображення результатів відновлених сигналів
fig, axes = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))

for i, ax in enumerate(axes.flat):
    ax.plot(time_values, restored_signals[i], linewidth=1)
    ax.set_title(f'Відновлений сигнал (Dt={discretization_steps[i]})')
    ax.set_xlabel('Час (с)')
    ax.set_ylabel('Значення сигналу')
    ax.grid(True)

fig.suptitle('Відновлені сигнали', fontsize=14)
fig.supxlabel('Час (с)', fontsize=14)
fig.supylabel('Значення сигналу', fontsize=14)
plt.tight_layout()
plt.savefig(output_directory + 'restored_signals.png', dpi=600)
plt.show()

# Побудова залежності різниці дисперсій від кроку дискретизації
plt.figure(figsize=(8, 6))
plt.plot(discretization_steps, variances, marker='o', linestyle='-')
plt.title('Залежність різниці дисперсій від кроку дискретизації')
plt.xlabel('Крок дискретизації')
plt.ylabel('Різниця дисперсій')
plt.grid(True)
plt.savefig(output_directory + 'variance_difference.png', dpi=600)
plt.show()

# Побудова залежності співвідношення сигнал-шум від кроку дискретизації
plt.figure(figsize=(8, 6))
plt.plot(discretization_steps, snr_ratios, marker='o', linestyle='-')
plt.title('Залежність співвідношення сигнал-шум від кроку дискретизації')
plt.xlabel('Крок дискретизації')
plt.ylabel('Співвідношення сигнал-шум')
plt.grid(True)
plt.savefig(output_directory + 'snr_ratio.png', dpi=600)
plt.show()

