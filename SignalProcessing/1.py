import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Функція для генерації випадкового сигналу
def r_signal(mean, std_dev, num_elements):
    return np.random.normal(mean, std_dev, num_elements)

# Функція для відображення та збереження сигналу
def plot_and_save_signal(time_values, signal_values, title, x_label, y_label, save_path=None):
    plt.figure(figsize=(21, 14))
    plt.plot(time_values, signal_values, linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()

# Створення папки для збереження
output_directory = './save/'
os.makedirs(output_directory, exist_ok=True)

# Параметри сигналу
n = 500  # Довжина сигналу (у відліках)
Fs = 1000  # Частота дискретизації (Гц)
F_max = 31  # Максимальна частота сигналу (Гц)

# Генерація випадкового сигналу
r_signal = r_signal(0, 10, n)

# Визначення відліків часу
time_values = np.arange(n) / Fs

# Розрахунок параметрів фільтру
w = F_max / (Fs / 2)
order = 3  # Порядок фільтру
sos = signal.butter(order, w, 'low', output='sos')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(sos, r_signal)

# Відображення та збереження результатів
output_path_before = os.path.join(output_directory, f'random_signal.png')
output_path_after = os.path.join(output_directory, f'filtered_signal.png')

plot_and_save_signal(time_values, r_signal, 'Випадковий сигнал (до фільтрації)', 'Час, сек', 'Значення сигналу',
            output_path_before)
plot_and_save_signal(time_values, filtered_signal, f'Сигнал  F_max = {F_max} Гц', 'Час, сек',
            'Значення після фільтрації', output_path_after)
