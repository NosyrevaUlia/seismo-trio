import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert
import io
import segyio
import tempfile
import os

# ============================================================
# 1. ЗАГОЛОВОК СТРАНИЦЫ
# ============================================================
st.set_page_config(page_title="Генератор сейсмических трасс", layout="wide")
st.title("Генератор синтетических сейсмических трасс")
st.markdown("Настройте параметры геологической модели и получите синтетическую трассу в реальном времени.")


# ============================================================
# 2.1 ФУНКЦИЯ СОХРАНЕНИЯ В ФОРМАТ SEG-Y
# ============================================================
def save_traces_to_sgy(traces, dt_ms, filename="synthetic_section.sgy",
                       trace_numbers=None, title="Synthetic Section"):
    """
    Сохранение массива трасс в SEG-Y файл.
    ----------
    traces : np.ndarray                                  массив трасс размером (num_traces, num_samples)
    dt_ms : float                                        интервал дискретизации в миллисекундах
    filename : str                                       имя выходного файла
    trace_numbers : list или np.ndarray, optional        номера трасс (если None, будут 1, 2, 3, ...)
    title : str                                          текстовое описание для заголовка
    ----------
    filename : str                                       имя созданного файла
    """

    num_traces, num_samples = traces.shape
    dt_us = int(dt_ms * 1000)

    # Создание спецификации файла
    spec = segyio.spec()
    spec.ilines = range(1, num_traces + 1)
    spec.xlines = [1]
    spec.samples = range(num_samples)
    spec.format = 5

    with segyio.create(filename, spec) as f:

        # Бинарный заголовок
        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Samples] = num_samples
        f.bin[segyio.BinField.Format] = 5

        # Текстовый заголовок
        text_header = []
        text_header.append(f"C 1 {title}")
        text_header.append(f"C 2 Created by synthetic seismic generator")
        text_header.append(f"C 3 Number of traces: {num_traces}")
        text_header.append(f"C 4 Samples per trace: {num_samples}")
        text_header.append(f"C 5 Sample interval: {dt_ms} ms ({dt_us} us)")
        text_header.append(f"C 6 Data format: IEEE 32-bit float")
        text_header.append(f"C 7 Date: {np.datetime64('today')}")

        while len(text_header) < 40:
            text_header.append(" " * 80)

        text_bytes = "\n".join(text_header)[:3200].encode('ascii', errors='ignore')
        f.text[0] = text_bytes.ljust(3200, b' ')

        # Трассы и заголовки
        for i in range(num_traces):
            trace_num = i + 1 if trace_numbers is None else trace_numbers[i]

            # Данные трассы
            f.trace[i] = traces[i].astype(np.float32)

            # Заголовок трассы
            f.header[i][segyio.TraceField.TraceNumber] = trace_num
            f.header[i][segyio.TraceField.TraceIdentifier] = 1
            f.header[i][segyio.TraceField.CDP] = trace_num
            f.header[i][segyio.TraceField.CDP_TRACE] = i + 1
            f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = i + 1
            f.header[i][segyio.TraceField.TRACE_SEQUENCE_FILE] = i + 1

    return filename


# ============================================================
# 2.2 КЛАССЫ ДЛЯ ГЕНЕРАЦИИ
# ============================================================

class RickerWaveletGenerator:
    """Генератор вейвлета Рикера."""

    def __init__(self, wavelet_freq=30, dt=0.002, length=0.128):
        self.wavelet_freq = wavelet_freq
        self.dt = dt
        self.length = length
        self.wavelet_time, self.wavelet = self._create_ricker_wavelet()

    def _create_ricker_wavelet(self):
        t = np.arange(-self.length / 2, self.length / 2, self.dt)
        f = self.wavelet_freq
        tau = np.pi * f * t
        wavelet = (1 - 2 * tau ** 2) * np.exp(-tau ** 2)
        return t, wavelet

    def get_wavelet(self):
        return self.wavelet_time, self.wavelet


class ReflectionCoefficients:
    """Расчёт коэффициентов отражения."""

    def __init__(self, depths, velocities, densities, angle=0):
        self.depths = np.array(depths)
        self.velocities = np.array(velocities)
        self.densities = np.array(densities)
        self.angle = np.radians(angle)  # перевод в радианы
        self.impedances = self.densities * self.velocities
        self.reflection_coeffs, self.two_way_times = self._compute_reflection()

    def _compute_reflection(self):
        R = []
        times = []
        cumulative = 0.0
        cos_angle = np.cos(self.angle)

        for i in range(len(self.depths)):
            Z_upper = self.impedances[i]
            Z_lower = self.impedances[i + 1]
            coeff = (Z_lower - Z_upper) / (Z_lower + Z_upper)
            R.append(coeff)
            if i == 0:
                thickness = self.depths[0]
            else:
                thickness = self.depths[i] - self.depths[i - 1]
            # Время с учётом угла наклона (увеличение пути)
            cumulative += 2 * thickness / (self.velocities[i] * cos_angle)
            times.append(cumulative)
        return np.array(R), np.array(times)

    def get_reflectivity_series(self, dt, total_time):
        num_samples = int(total_time / dt) + 1
        reflectivity = np.zeros(num_samples)
        for t, r in zip(self.two_way_times, self.reflection_coeffs):
            idx = int(round(t / dt))
            if idx < num_samples:
                reflectivity[idx] = r
        return reflectivity


class SeismicTraceGenerator:
    """Генератор синтетической трассы."""

    def __init__(self, wavelet_freq=30, dt=0.002, noise_level=0.0, amplitude=1000):
        self.wavelet_freq = wavelet_freq
        self.dt = dt
        self.noise_level = noise_level
        self.amplitude = amplitude
        self.total_time = None
        self.time = None
        wavelet_gen = RickerWaveletGenerator(wavelet_freq=wavelet_freq, dt=dt, length=0.2)
        self.wavelet_time, self.wavelet = wavelet_gen.get_wavelet()

    def set_total_time(self, total_time):
        self.total_time = total_time
        self.time = np.arange(0, total_time, self.dt)

    def generate_from_reflectivity(self, reflectivity):
        """Генерация трассы из временного ряда коэффициентов отражения."""
        trace_full = np.convolve(reflectivity, self.wavelet, mode='full')
        trace = trace_full[:len(reflectivity)]
        trace = trace * self.amplitude

        if self.noise_level > 0:
            rms = np.sqrt(np.mean(trace ** 2))
            if rms > 0:
                noise = np.random.randn(len(trace)) * self.noise_level * rms
                trace = trace + noise

        return trace

    def generate_from_model(self, depths, velocities, densities, total_time, angle=0, trace_length=None):
        rc = ReflectionCoefficients(depths, velocities, densities, angle)
        reflectivity = rc.get_reflectivity_series(self.dt, total_time)

        time_axis = np.arange(0, total_time + self.dt, self.dt)
        time_axis = time_axis[:len(reflectivity)]

        trace = self.generate_from_reflectivity(reflectivity)

        if trace_length is not None:
            if len(trace) > trace_length:
                trace = trace[:trace_length]
                time_axis = time_axis[:trace_length]
            elif len(trace) < trace_length:
                pad_len = trace_length - len(trace)
                trace = np.pad(trace, (0, pad_len), 'constant')
                time_axis = np.arange(0, trace_length * self.dt, self.dt)

        return time_axis, trace, rc


# ============================================================
# 3. ГИБРИДНЫЙ ГЕНЕРАТОР
# ============================================================

class HybridGenerator:
    """Гибридный генератор, объединяющий Монте-Карло и микрослоистость"""

    def __init__(self, wavelet_freq=30, dt=0.002, total_time=3.0, noise_level=0.0, amplitude=1000):
        self.wavelet_freq = wavelet_freq
        self.dt = dt
        self.total_time = total_time
        self.noise_level = noise_level
        self.amplitude = amplitude
        self.num_samples = int(total_time / dt) + 1
        self.time_axis = np.arange(0, total_time + dt, dt)[:self.num_samples]

        self.trace_gen = SeismicTraceGenerator(
            wavelet_freq=wavelet_freq,
            dt=dt,
            noise_level=noise_level,
            amplitude=amplitude
        )
        self.trace_gen.set_total_time(total_time)

    def set_total_time(self, total_time):
        self.total_time = total_time
        self.num_samples = int(total_time / self.dt) + 1
        self.time_axis = np.arange(0, total_time + self.dt, self.dt)[:self.num_samples]
        self.trace_gen.set_total_time(total_time)

    @staticmethod
    def sample_geology(distributions, enforce_sort=True):
        """Генерация одного случайного набора геологических параметров."""
        depths = []
        velocities = []
        densities = []

        for d in distributions.get('depths', []):
            dist_type = d.get('type', 'fixed')
            if dist_type == 'uniform':
                h = np.random.uniform(d['low'], d['high'])
            elif dist_type == 'normal':
                h = np.random.normal(d['mu'], d['sigma'])
            elif dist_type == 'lognormal':
                h = np.random.lognormal(d['mu'], d['sigma'])
            else:
                h = d.get('value', 1000)
            depths.append(h)

        for v in distributions.get('velocities', []):
            dist_type = v.get('type', 'fixed')
            if dist_type == 'normal':
                val = np.random.normal(v['mu'], v['sigma'])
            elif dist_type == 'uniform':
                val = np.random.uniform(v['low'], v['high'])
            elif dist_type == 'lognormal':
                val = np.random.lognormal(v['mu'], v['sigma'])
            else:
                val = v.get('value', 2500)
            val = max(val, 500)
            velocities.append(val)

        for r in distributions.get('densities', []):
            dist_type = r.get('type', 'fixed')
            if dist_type == 'normal':
                val = np.random.normal(r['mu'], r['sigma'])
            elif dist_type == 'uniform':
                val = np.random.uniform(r['low'], r['high'])
            elif dist_type == 'lognormal':
                val = np.random.lognormal(r['mu'], r['sigma'])
            else:
                val = r.get('value', 2200)
            val = max(val, 1500)
            densities.append(val)

        depths = np.array(depths)
        if enforce_sort and len(depths) > 1:
            depths = np.sort(depths)

        return np.array(depths), np.array(velocities), np.array(densities)

    def _get_deterministic_model(self, distributions):
        """Получает детерминированную модель (средние значения параметров)."""
        depths = []
        for d in distributions.get('depths', []):
            if d.get('type') == 'uniform':
                depths.append((d['low'] + d['high']) / 2)
            elif d.get('type') == 'normal':
                depths.append(d['mu'])
            else:
                depths.append(d.get('value', 1000))

        velocities = []
        for v in distributions.get('velocities', []):
            if v.get('type') == 'normal':
                velocities.append(v['mu'])
            elif v.get('type') == 'uniform':
                velocities.append((v['low'] + v['high']) / 2)
            else:
                velocities.append(v.get('value', 2500))

        densities = []
        for r in distributions.get('densities', []):
            if r.get('type') == 'normal':
                densities.append(r['mu'])
            elif r.get('type') == 'uniform':
                densities.append((r['low'] + r['high']) / 2)
            else:
                densities.append(r.get('value', 2200))

        return np.array(depths), np.array(velocities), np.array(densities)

    def _compute_reflectivity_with_trends(self, depths, velocities, densities, layer_trends, angle=0):
        """Расчёт коэффициентов отражения с учётом микрослоистости внутри КАЖДОГО слоя."""
        reflectivity = np.zeros(self.num_samples)
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)

        for i in range(len(depths) + 1):
            if i == 0:
                top = 0
            else:
                top = depths[i - 1]

            if i == len(depths):
                if len(depths) > 0:
                    bottom = depths[-1] + 100
                else:
                    continue
            else:
                bottom = depths[i]

            if bottom - top <= 0:
                continue

            V0 = velocities[i]
            rho0 = densities[i]

            if i < len(layer_trends):
                trend = layer_trends[i]
            else:
                trend = {'type': 'none'}

            if trend.get('type') == 'none':
                continue

            num_sublayers = 80
            z_micro = np.linspace(top, bottom, num_sublayers + 1)

            if trend.get('type') == 'linear':
                k = trend.get('k', 100)
                V_micro = V0 + k * (z_micro - top)
                rho_micro = rho0 + 0.1 * k * (z_micro - top)
            elif trend.get('type') == 'sinusoidal':
                A = trend.get('A', 150)
                L = trend.get('L', 50)
                V_micro = V0 + A * np.sin(2 * np.pi * (z_micro - top) / L)
                rho_micro = rho0 + 10 * np.sin(2 * np.pi * (z_micro - top) / L)
            elif trend.get('type') == 'random':
                sigma = trend.get('sigma', 50)
                noise = np.random.normal(0, sigma, len(z_micro))
                window = 5
                kernel = np.ones(window) / window
                noise_smooth = np.convolve(noise, kernel, mode='same')
                V_micro = V0 + noise_smooth
                rho_micro = rho0 + 0.1 * noise_smooth
            else:
                continue

            V_micro = np.clip(V_micro, 500, 8000)
            rho_micro = np.clip(rho_micro, 1500, 3500)

            Z_micro = V_micro * rho_micro

            for j in range(len(Z_micro) - 1):
                if Z_micro[j] > 0 and Z_micro[j + 1] > 0:
                    R = (Z_micro[j + 1] - Z_micro[j]) / (Z_micro[j + 1] + Z_micro[j])
                else:
                    R = 0
                if abs(R) > 0.99:
                    R = np.sign(R) * 0.99

                avg_v = np.mean(V_micro[:j + 2])
                t = 2 * (z_micro[j + 1] - top) / (avg_v * cos_angle) + 2 * top / (V0 * cos_angle)
                idx = int(round(t / self.dt))
                if idx < self.num_samples:
                    reflectivity[idx] += R

        return reflectivity

    def generate_trace(self, distributions, layer_trends,
                       use_mc=True, use_trends=True, angle=0,
                       seed=None, verbose=False):
        """Генерация одной трассы с учётом обоих эффектов."""
        if seed is not None:
            np.random.seed(seed)

        if use_mc:
            depths, velocities, densities = self.sample_geology(distributions)
        else:
            depths, velocities, densities = self._get_deterministic_model(distributions)

        if use_trends:
            reflectivity = self._compute_reflectivity_with_trends(
                depths, velocities, densities, layer_trends, angle
            )
        else:
            rc = ReflectionCoefficients(depths, velocities, densities, angle)
            reflectivity = rc.get_reflectivity_series(self.dt, self.total_time)

        trace = self.trace_gen.generate_from_reflectivity(reflectivity)

        if len(trace) > self.num_samples:
            trace = trace[:self.num_samples]
        elif len(trace) < self.num_samples:
            pad_len = self.num_samples - len(trace)
            trace = np.pad(trace, (0, pad_len), 'constant')

        mode = "Детерминированная модель"
        if use_mc and use_trends:
            mode = "ГИБРИД (Монте-Карло + Микрослоистость)"
        elif use_mc:
            mode = "Только Монте-Карло"
        elif use_trends:
            mode = "Только микрослоистость"
        else:
            mode = "Детерминированная модель"

        return {
            'trace': trace,
            'time_axis': self.time_axis,
            'depths': depths,
            'velocities': velocities,
            'densities': densities,
            'mode': mode,
            'reflectivity': reflectivity
        }


# ============================================================
# 4. БОКОВАЯ ПАНЕЛЬ С ПАРАМЕТРАМИ
# ============================================================
st.sidebar.header("⚙️ Параметры модели")

# ЧЕКБОКСЫ ДЛЯ ГИБРИДНОГО ГЕНЕРАТОРА
st.sidebar.subheader("Режимы генерации")
use_mc = st.sidebar.checkbox("Учитывать неопределённость (Монте-Карло)", value=False)
use_trends = st.sidebar.checkbox("Учитывать неоднородность слоёв", value=False)

# Длина трассы
st.sidebar.subheader("Параметры трассы")
trace_length = st.sidebar.slider("Длина трассы (количество отсчетов)", 100, 10000, 2000)

# Количество горизонтов
num_horizons = st.sidebar.slider("Количество отражающих горизонтов", 1, 20, 3)

# ============================================================
# ИЗМЕНЕНИЕ 1: Шаг дискретизации — выпадающий список (0.5, 1, 2, 4 мс)
# ============================================================
dt_options = [0.5, 1.0, 2.0, 4.0]
dt_ms = st.sidebar.selectbox("Интервал дискретизации dt (мс)", dt_options, index=1)
dt = dt_ms / 1000

# Длительность записи (до 5 секунд)
total_time = st.sidebar.number_input("Длительность записи (с)",
                                     min_value=0.5, max_value=5.0, value=5.0, step=0.5)

# ============================================================
# ИЗМЕНЕНИЕ 2: Автоматический расчёт количества отсчётов
# ============================================================
num_samples_auto = int(total_time / dt) + 1
st.sidebar.info(f"Количество отсчетов: {num_samples_auto} (рассчитано автоматически)")

# ============================================================
# ИЗМЕНЕНИЕ 3: Угол наклона слоёв
# ============================================================
angle = st.sidebar.slider("Угол наклона слоёв (градусы)", 0, 60, 0, 1,
                          help="Угол наклона отражающих границ. 0° — горизонтальные слои.")

# Количество трасс
num_traces_to_generate = st.sidebar.number_input(
    "Количество трасс для генерации",
    min_value=1,
    max_value=50000,
    value=100,
    step=1
)

# Параметры вейвлета
wavelet_freq = st.sidebar.slider("Частота вейвлета (Гц)", 5, 500, 30)

# Уровень шума
noise_std = st.sidebar.slider("Уровень шума", 0.0, 1.0, 0.0, 0.01)

# Амплитуда
amplitude = st.sidebar.slider("Амплитуда", 50, 2500, 1000, 50)

# ============================================================
# 5. ГЕОЛОГИЧЕСКАЯ МОДЕЛЬ
# ============================================================
st.sidebar.subheader("Геологическая модель")

MAX_DEPTH = 12000


def generate_default_values(num_horizons):
    """Генерирует значения по умолчанию для заданного количества горизонтов"""
    depths = []
    velocities = []
    densities = []

    depths.append(300)
    velocities.append(1000)
    densities.append(2200)

    for i in range(num_horizons):
        depth = 500 + (i + 1) * ((MAX_DEPTH - 1000) // num_horizons)
        depth = min(depth, MAX_DEPTH - 200)
        depths.append(depth)

        vel = 2200 + (i + 1) * 150
        velocities.append(min(vel, 6000))

        rho = 2200 + (i + 1) * 60
        densities.append(min(rho, 3500))

    return depths, velocities, densities


default_depths, default_velocities, default_densities = generate_default_values(20)

depths = []
velocities = []
densities = []

# Слой 0
velocities.append(st.sidebar.number_input(
    "Скорость слоя 0 (м/с)",
    min_value=500, max_value=6000, value=default_velocities[0], step=50
))
densities.append(st.sidebar.number_input(
    "Плотность слоя 0 (кг/м³)",
    min_value=1500, max_value=3500, value=default_densities[0], step=50
))

# Горизонты и слои
layer_trends = []
for i in range(num_horizons):
    st.sidebar.markdown(f"---")
    st.sidebar.markdown(f"**Горизонт {i + 1}**")

    depth_val = default_depths[i + 1] if i + 1 < len(default_depths) else 500 + i * 500
    depth = st.sidebar.number_input(
        f"Глубина {i + 1} (м)",
        min_value=100, max_value=MAX_DEPTH,
        value=depth_val, step=50
    )
    depths.append(depth)

    v_val = default_velocities[i + 1] if i + 1 < len(default_velocities) else 2000 + (i + 1) * 300
    v = st.sidebar.number_input(
        f"Скорость слоя {i + 1} (м/с)",
        min_value=500, max_value=6000, value=v_val, step=50
    )
    velocities.append(v)

    rho_val = default_densities[i + 1] if i + 1 < len(default_densities) else 2200 + i * 100
    rho = st.sidebar.number_input(
        f"Плотность слоя {i + 1} (кг/м³)",
        min_value=1500, max_value=3500, value=rho_val, step=50
    )
    densities.append(rho)

    # Тренды для каждого слоя
    st.sidebar.markdown(f"**Тренд слоя {i + 1}**")
    trend_type = st.sidebar.selectbox(
        f"Тип тренда {i + 1}",
        ['none', 'linear', 'sinusoidal', 'random'],
        index=0,
        key=f"trend_{i}"
    )
    if trend_type != 'none':
        trend_params = {}
        if trend_type == 'linear':
            trend_params['k'] = st.sidebar.number_input(f"Градиент k {i + 1}", 10, 500, 100, key=f"k_{i}")
        elif trend_type == 'sinusoidal':
            trend_params['A'] = st.sidebar.number_input(f"Амплитуда A {i + 1}", 10, 500, 150, key=f"A_{i}")
            trend_params['L'] = st.sidebar.number_input(f"Период L {i + 1}", 10, 200, 50, key=f"L_{i}")
        elif trend_type == 'random':
            trend_params['sigma'] = st.sidebar.number_input(f"Sigma {i + 1}", 10, 200, 50, key=f"sigma_{i}")
        layer_trends.append({'type': trend_type, **trend_params})
    else:
        layer_trends.append({'type': 'none'})

# ============================================================
# 6. ГЕНЕРАЦИЯ И ОТОБРАЖЕНИЕ
# ============================================================
try:
    # Гибридный генератор с амплитудой
    hybrid_gen = HybridGenerator(
        wavelet_freq=wavelet_freq,
        dt=dt,
        total_time=total_time,
        noise_level=noise_std,
        amplitude=amplitude
    )

    # Распределения для Монте-Карло
    distributions = {
        'velocities': [{'type': 'normal', 'mu': v, 'sigma': v * 0.05} for v in velocities],
        'densities': [{'type': 'normal', 'mu': r, 'sigma': r * 0.03} for r in densities],
        'depths': [{'type': 'uniform', 'low': d * 0.9, 'high': d * 1.1} for d in depths]
    }

    # Генерируем трассу через гибридный генератор с углом
    result = hybrid_gen.generate_trace(
        distributions, layer_trends,
        use_mc=use_mc,
        use_trends=use_trends,
        angle=angle,  # Передаём угол
        seed=42
    )

    trace = result['trace']
    time_axis = result['time_axis']
    mode = result['mode']

    st.header("Результат генерации")
    st.info(
        f"Режим: **{mode}** | Длина трассы: {len(trace)} отсчетов | Время записи: {time_axis[-1]:.2f} с | Угол наклона: {angle}°")

    # Основной график
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, trace, 'b-', linewidth=1.5, label='Синтетическая трасса')

    # Отмечаем горизонты с учётом угла
    rc = ReflectionCoefficients(result['depths'], result['velocities'], result['densities'], angle)
    for i, t in enumerate(rc.two_way_times):
        if t < time_axis[-1]:
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.7,
                       label=f'Горизонт {i + 1}: t={t:.3f}c, R={rc.reflection_coeffs[i]:.3f}' if i == 0 else "")

    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Амплитуда')
    ax.set_title(f'Синтетическая сейсмическая трасса ({mode})')
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # Генерация ансамбля трасс для вертикального отображения
    if num_traces_to_generate > 1:
        st.subheader("Ансамбль синтетических трасс")

        with st.spinner(f"Генерация {num_traces_to_generate} трасс..."):
            traces_ensemble = []

            # Параметры из уже сгенерированной трассы
            base_depths = result['depths']
            base_velocities = result['velocities']
            base_densities = result['densities']

            # Генерируем множество трасс
            for i in range(num_traces_to_generate):
                if use_mc:
                    # Монте-Карло: каждая трасса со СЛУЧАЙНЫМИ параметрами
                    depths_i, velocities_i, densities_i = HybridGenerator.sample_geology(distributions)

                    # Тренды с вариациями для Монте-Карло
                    if use_trends:
                        layer_trends_i = []
                        for trend in layer_trends:
                            if trend['type'] == 'linear':
                                trend_i = trend.copy()
                                trend_i['k'] = trend['k'] * (1 + np.random.randn() * 0.1)
                            elif trend['type'] == 'sinusoidal':
                                trend_i = trend.copy()
                                trend_i['A'] = trend['A'] * (1 + np.random.randn() * 0.1)
                                trend_i['L'] = trend['L'] * (1 + np.random.randn() * 0.1)
                            elif trend['type'] == 'random':
                                trend_i = trend.copy()
                                trend_i['sigma'] = trend['sigma'] * (1 + np.random.randn() * 0.1)
                            else:
                                trend_i = trend.copy()
                            layer_trends_i.append(trend_i)
                    else:
                        layer_trends_i = layer_trends

                    # Генерируем трассу со случайным seed
                    result_i = hybrid_gen.generate_trace(
                        distributions, layer_trends_i,
                        use_mc=use_mc,
                        use_trends=use_trends,
                        angle=angle,  # Передаём угол
                        seed=None
                    )
                else:
                    # Детерминированный режим
                    result_i = hybrid_gen.generate_trace(
                        distributions, layer_trends,
                        use_mc=use_mc,
                        use_trends=use_trends,
                        angle=angle,  # Передаём угол
                        seed=42
                    )

                traces_ensemble.append(result_i['trace'])

            traces_ensemble = np.array(traces_ensemble)

        # Создаем вертикальный график ансамбля (как в SEG-Y)
        fig_ensemble, ax_ensemble = plt.subplots(figsize=(12, 10))

        # Определяем масштаб для отображения
        max_amplitude = np.max(np.abs(traces_ensemble))
        if max_amplitude > 0:
            # Нормализуем трассы
            normalized_traces = traces_ensemble / max_amplitude

            # Отображаем трассы вертикально (как в сейсмических разрезах)
            trace_spacing = 1.0
            num_show = min(len(traces_ensemble), 200)

            # Определяем режим для подписи
            if use_mc:
                plot_title = f'Ансамбль трасс (Монте-Карло, n={len(traces_ensemble)})'
                line_color = 'k-'
                line_width = 0.6
                alpha = 0.5
            else:
                plot_title = f'Ансамбль трасс (Детерминированный, n={len(traces_ensemble)})'
                line_color = 'k-'
                line_width = 0.8
                alpha = 0.7

            # Для каждой трассы: X = номер трассы, Y = время, амплитуда = смещение по X
            for i in range(num_show):
                # Смещение трассы по X (номер трассы)
                x_offset = i * trace_spacing
                # Амплитуда добавляется/вычитается от центральной линии
                x_positions = x_offset + normalized_traces[i] * 0.8
                # Время идет по оси Y (вертикально)
                y_positions = time_axis

                ax_ensemble.plot(
                    x_positions,
                    y_positions,
                    line_color,
                    linewidth=line_width,
                    alpha=alpha
                )

            # Настройки графика (как в сейсмических разрезах)
            ax_ensemble.set_xlabel('Номер трассы')
            ax_ensemble.set_ylabel('Время (с)')
            ax_ensemble.set_title(plot_title)
            ax_ensemble.grid(True, alpha=0.2)

            # Инвертируем ось Y (время возрастает вниз, как в сейсмике)
            ax_ensemble.invert_yaxis()

            # Настройка осей
            ax_ensemble.set_xlim(-1.5, num_show * trace_spacing + 1.5)
            ax_ensemble.set_ylim(time_axis[-1], 0)  # Время сверху вниз

            # Добавляем отметки горизонтальных линий (горизонты) с учётом угла
            rc_ensemble = ReflectionCoefficients(result['depths'], result['velocities'], result['densities'], angle)
            for t in rc_ensemble.two_way_times:
                if t < time_axis[-1]:
                    ax_ensemble.axhline(y=t, color='red', linestyle='--', alpha=0.5, linewidth=1)

            # Отмечаем номера трасс на оси X
            if num_show <= 50:
                x_ticks = np.arange(0, num_show * trace_spacing, trace_spacing * max(1, num_show // 10))
                x_labels = [f"{int(i)}" for i in x_ticks / trace_spacing]
                ax_ensemble.set_xticks(x_ticks)
                ax_ensemble.set_xticklabels(x_labels)

            st.pyplot(fig_ensemble)
            plt.close(fig_ensemble)

            # Информация о режиме
            if use_mc:
                st.caption(f"Монте-Карло: {len(traces_ensemble)} уникальных трасс со случайными параметрами")
            else:
                st.caption(f"Детерминированный режим: {len(traces_ensemble)} идентичных трасс")

    plt.close(fig)

    # Вейвлет и коэффициенты
    col1, col2 = st.columns(2)

    with col1:
        wav_time, wav_vals = hybrid_gen.trace_gen.wavelet_time, hybrid_gen.trace_gen.wavelet
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(wav_time, wav_vals)
        ax2.set_title(f'Вейвлет Рикера, f={wavelet_freq} Гц')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Амплитуда')
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        times_show = [t for t in rc.two_way_times if t < time_axis[-1]]
        coeffs_show = rc.reflection_coeffs[:len(times_show)]
        if len(times_show) > 0:
            ax3.stem(times_show, coeffs_show, linefmt='r-', markerfmt='ro', basefmt='k-')
        ax3.set_title('Коэффициенты отражения')
        ax3.set_xlabel('Время (с)')
        ax3.set_ylabel('R')
        ax3.grid(True)
        st.pyplot(fig3)
        plt.close(fig3)

    # Спектр
    st.subheader("Спектральный анализ")
    col3, col4 = st.columns(2)

    with col3:
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        freq = np.fft.rfftfreq(len(trace), d=dt)
        spectrum = np.abs(np.fft.rfft(trace))
        ax4.plot(freq, spectrum)
        ax4.set_xlim(0, 100)
        ax4.set_xlabel('Частота (Гц)')
        ax4.set_ylabel('Амплитуда')
        ax4.set_title('Амплитудный спектр')
        ax4.grid(True)
        st.pyplot(fig4)
        plt.close(fig4)

    with col4:
        analytic = hilbert(trace)
        envelope = np.abs(analytic)
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        ax5.plot(time_axis, envelope)
        ax5.set_title('Огибающая трассы')
        ax5.set_xlabel('Время (с)')
        ax5.set_ylabel('Амплитуда')
        ax5.grid(True)
        st.pyplot(fig5)
        plt.close(fig5)

    # Таблицы
    st.header("Параметры модели")

    model_data = []
    for i in range(len(result['depths']) + 1):
        model_data.append({
            "Слой": i,
            "V (м/с)": result['velocities'][i],
            "ρ (кг/м³)": result['densities'][i],
            "Z (ρ·V)": result['velocities'][i] * result['densities'][i]
        })
    st.table(model_data)

    boundaries_data = []
    for i in range(len(result['depths'])):
        if rc.two_way_times[i] < time_axis[-1]:
            boundaries_data.append({
                "Горизонт": i + 1,
                "Глубина (м)": result['depths'][i],
                "Время (с)": f"{rc.two_way_times[i]:.3f}",
                "R": f"{rc.reflection_coeffs[i]:.4f}"
            })
    if boundaries_data:
        st.table(boundaries_data)

    # Экспорт
    st.subheader("💾 Экспорт")
    df = pd.DataFrame({"Время (с)": time_axis, "Амплитуда": trace})
    csv = df.to_csv(index=False)
    st.download_button("Скачать трассу (CSV)", csv, "trace.csv", "text/csv")

    np_buffer = io.BytesIO()
    np.save(np_buffer, trace)
    st.download_button("Скачать трассу (NPY)", np_buffer.getvalue(), "trace.npy", "application/octet-stream")

    # Экспорт всех трасс в SEG-Y
    if 'traces_ensemble' in locals() and traces_ensemble is not None and len(traces_ensemble) > 0:
        sgy_buffer = io.BytesIO()

        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as tmp_file:
            tmp_filename = tmp_file.name

        try:
            # Сохраняем ансамбль трасс в SEG-Y
            save_traces_to_sgy(
                traces=traces_ensemble,
                dt_ms=dt * 1000,
                filename=tmp_filename,
                title=f"Синтетический разрез ({mode})"
            )

            # Читаем файл в буфер
            with open(tmp_filename, 'rb') as f:
                sgy_buffer = io.BytesIO(f.read())

            st.download_button(
                label=f"Скачать все трассы SEG-Y",
                data=sgy_buffer.getvalue(),
                file_name="generated_traces.sgy",
                mime="application/octet-stream"
            )
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

except Exception as e:
    st.error(f"Ошибка: {e}")
    st.markdown("Проверьте корректность введённых параметров (глубины должны возрастать).")