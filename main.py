from obspy import Stream, Trace
from obspy.io.segy.core import _read_segy, _write_segy
import numpy as np

# Загружаем SEGY файл
segy_file_path = '001_forKORF55_3902.sgy'
filename_out = "result_11_1_200.sgy"
segy_file = _read_segy(segy_file_path)
traces = np.array([trace.data for trace in segy_file])
window_size = 200  # размер окна сглаживания по времени (в мс)
T = int(window_size)  # преобразуем это значение в соответствующий формат
X = 11  # размер по Y
Y = 0  # размер по X
n_samples = len(segy_file[0].data)

# Функция для расчета средней амплитуды по заданной формуле
def calculate_average_amplitude(trace_data, T):
    half_T = T // 2
    n = len(trace_data)
    A = np.zeros(n)

    for t in range(n):
        start_index = max(0, t - half_T)
        end_index = min(n, t + half_T + 1)  # include end_index

        # Расчет средней амплитуды
        A[t] = np.sum(np.abs(trace_data[start_index:end_index])) / (end_index - start_index)

    return A

# Рассчитываем среднюю амплитуду для каждой трассы
average_amplitudes = np.array([calculate_average_amplitude(trace, T) for trace in traces])

# Рассчитываем Amid для каждой трассы
Amid = np.zeros((len(traces), n_samples))

for tr in range(len(traces)):
    for t in range(n_samples):
        sum_amplitudes = 0
        count = 0

        # Учитываем соседние трассы в пределах окна
        for n in range(-X // 2, X // 2 + 1):  # X по Y
            if (tr + n >= 0) and (tr + n < average_amplitudes.shape[0]):
                sum_amplitudes += average_amplitudes[tr + n, t]  # A(t,x-n,y-k)
                count += 1

        Amid[tr, t] = sum_amplitudes / count if count > 0 else 0  # Делим на количество найденных

# Корректировка амплитуды
for tr in range(len(segy_file)):
    trace_data = segy_file[tr].data
    norm_function = np.divide(Amid[tr], average_amplitudes[tr], out=np.zeros_like(Amid[tr]), where=average_amplitudes[tr] != 0)  # Рассчитываем нормирующую функцию

    traces[tr] *= norm_function  # Корректируем амплитуды
    segy_file[tr].data = traces[tr]  # Обновляем данные в оригинальном SEGY файле

# Сохраняем скорректированные трассы
_write_segy(segy_file, filename_out)