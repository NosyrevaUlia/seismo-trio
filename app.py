import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert
import io

# ============================================================
# 1. ЗАГОЛОВОК СТРАНИЦЫ
# ============================================================
st.set_page_config(page_title="Генератор сейсмических трасс", layout="wide")
st.title("🌍 Генератор синтетических сейсмических трасс")
st.markdown("Настройте параметры геологической модели и получите синтетическую трассу в реальном времени.")


# ============================================================
# 2. КЛАССЫ ДЛЯ ГЕНЕРАЦИИ
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

    def __init__(self, depths, velocities, densities):
        self.depths = np.array(depths)
        self.velocities = np.array(velocities)
        self.densities = np.array(densities)
        self.impedances = self.densities * self.velocities
        self.reflection_coeffs, self.two_way_times = self._compute_reflection()

    def _compute_reflection(self):
        R = []
        times = []
        cumulative = 0.0
        for i in range(len(self.depths)):
            Z_upper = self.impedances[i]
            Z_lower = self.impedances[i + 1]
            coeff = (Z_lower - Z_upper) / (Z_lower + Z_upper)
            R.append(coeff)
            if i == 0:
                thickness = self.depths[0]
            else:
                thickness = self.depths[i] - self.depths[i - 1]
            cumulative += 2 * thickness / self.velocities[i]
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
        wavelet_gen = RickerWaveletGenerator(wavelet_freq=wavelet_freq, dt=dt, length=0.2)
        self.wavelet_time, self.wavelet = wavelet_gen.get_wavelet()

    def generate_from_model(self, depths, velocities, densities, total_time, trace_length=None):
        rc = ReflectionCoefficients(depths, velocities, densities)
        reflectivity = rc.get_reflectivity_series(self.dt, total_time)

        time_axis = np.arange(0, total_time + self.dt, self.dt)
        time_axis = time_axis[:len(reflectivity)]

        trace_full = np.convolve(reflectivity, self.wavelet, mode='full')
        trace = trace_full[:len(reflectivity)]

        # Масштабирование с учетом амплитуды
        trace = trace * self.amplitude

        # Если задана длина трассы - обрезаем или дополняем нулями
        if trace_length is not None:
            if len(trace) > trace_length:
                trace = trace[:trace_length]
                time_axis = time_axis[:trace_length]
            elif len(trace) < trace_length:
                pad_len = trace_length - len(trace)
                trace = np.pad(trace, (0, pad_len), 'constant')
                time_axis = np.arange(0, trace_length * self.dt, self.dt)

        # Шум
        if self.noise_level > 0:
            rms = np.sqrt(np.mean(trace ** 2))
            if rms > 0:
                noise = np.random.randn(len(trace)) * self.noise_level * rms
                trace = trace + noise

        return time_axis, trace, rc


# ============================================================
# 3. БОКОВАЯ ПАНЕЛЬ С ПАРАМЕТРАМИ
# ============================================================
st.sidebar.header("⚙️ Параметры модели")

# Длина трассы (trace)
trace_length = st.sidebar.slider("Длина трассы (количество отсчетов)", 100, 10000, 2000)

# Количество горизонтов
num_horizons = st.sidebar.slider("Количество отражающих горизонтов", 1, 20, 3)

# Параметры дискретизации
dt = st.sidebar.number_input("Интервал дискретизации dt (мс)",
                             min_value=1.0, max_value=4.0, value=1.0) / 1000

# Длительность записи
total_time = st.sidebar.number_input("Длительность записи (с)",
                                     min_value=0.5, max_value=5.0, value=2.0)

# Параметры вейвлета
wavelet_freq = st.sidebar.slider("Частота вейвлета (Гц)", 5, 500, 30)

# Уровень шума
noise_std = st.sidebar.slider("Уровень шума", 0.0, 1.0, 0.05)

# Амплитуда
amplitude = st.sidebar.slider("Амплитуда", 50, 2500, 1000)

# ============================================================
# 4. ГЕОЛОГИЧЕСКАЯ МОДЕЛЬ
# ============================================================
st.sidebar.subheader("📊 Геологическая модель")

# ✅ Максимальная глубина 12000 м
MAX_DEPTH = 12000


# ✅ Генерируем значения по умолчанию
def generate_default_values(num_horizons):
    """Генерирует значения по умолчанию для заданного количества горизонтов"""
    depths = []
    velocities = []
    densities = []

    # Начальные значения
    depths.append(300)
    velocities.append(1000)
    densities.append(2200)

    # Генерируем для каждого горизонта
    for i in range(num_horizons):
        # Глубина: от 500 до MAX_DEPTH-500 с равномерным шагом
        depth = 500 + (i + 1) * ((MAX_DEPTH - 1000) // num_horizons)
        # Ограничиваем максимальной глубиной
        depth = min(depth, MAX_DEPTH - 200)
        depths.append(depth)

        # Скорость: увеличивается от 2200 до 5000
        vel = 2200 + (i + 1) * 150
        velocities.append(min(vel, 6000))

        # Плотность: увеличивается от 2200 до 3300
        rho = 2200 + (i + 1) * 60
        densities.append(min(rho, 3500))

    return depths, velocities, densities


# Генерируем значения по умолчанию
default_depths, default_velocities, default_densities = generate_default_values(20)

depths = []
velocities = []
densities = []

# Слой 0
velocities.append(st.sidebar.number_input(
    "Скорость слоя 0 (м/с)",
    min_value=500, max_value=6000, value=default_velocities[0]
))
densities.append(st.sidebar.number_input(
    "Плотность слоя 0 (кг/м³)",
    min_value=1500, max_value=3500, value=default_densities[0]
))

# Горизонты и слои
for i in range(num_horizons):
    st.sidebar.markdown(f"---")
    st.sidebar.markdown(f"**Горизонт {i + 1}**")

    depth_val = default_depths[i + 1] if i + 1 < len(default_depths) else 500 + i * 500
    depth = st.sidebar.number_input(
        f"Глубина {i + 1} (м)",
        min_value=100, max_value=MAX_DEPTH,  # ✅ 12000 м
        value=depth_val
    )
    depths.append(depth)

    v_val = default_velocities[i + 1] if i + 1 < len(default_velocities) else 2000 + (i + 1) * 300
    v = st.sidebar.number_input(
        f"Скорость слоя {i + 1} (м/с)",
        min_value=500, max_value=6000, value=v_val
    )
    velocities.append(v)

    rho_val = default_densities[i + 1] if i + 1 < len(default_densities) else 2200 + i * 100
    rho = st.sidebar.number_input(
        f"Плотность слоя {i + 1} (кг/м³)",
        min_value=1500, max_value=3500, value=rho_val
    )
    densities.append(rho)

# ============================================================
# 5. ГЕНЕРАЦИЯ И ОТОБРАЖЕНИЕ
# ============================================================
try:
    generator = SeismicTraceGenerator(
        wavelet_freq=wavelet_freq,
        dt=dt,
        noise_level=noise_std,
        amplitude=amplitude
    )
    time_axis, trace, rc_model = generator.generate_from_model(
        depths, velocities, densities, total_time, trace_length
    )

    st.header("📈 Результат генерации")

    # Информация о трассе
    st.info(f"Длина трассы: {len(trace)} отсчетов, Время записи: {time_axis[-1]:.2f} с")

    # Основной график
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, trace, 'b-', linewidth=1.5, label='Синтетическая трасса')
    for i, t in enumerate(rc_model.two_way_times):
        if t < time_axis[-1]:
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.7,
                       label=f'Горизонт {i + 1}: t={t:.3f}c, R={rc_model.reflection_coeffs[i]:.3f}' if i == 0 else "")
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Амплитуда')
    ax.set_title('Синтетическая сейсмическая трасса')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Вейвлет и коэффициенты
    col1, col2 = st.columns(2)

    with col1:
        wav_time, wav_vals = generator.wavelet_time, generator.wavelet
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
        times_show = [t for t in rc_model.two_way_times if t < time_axis[-1]]
        coeffs_show = rc_model.reflection_coeffs[:len(times_show)]
        if len(times_show) > 0:
            ax3.stem(times_show, coeffs_show, linefmt='r-', markerfmt='ro', basefmt='k-')
        ax3.set_title('Коэффициенты отражения')
        ax3.set_xlabel('Время (с)')
        ax3.set_ylabel('R')
        ax3.grid(True)
        st.pyplot(fig3)
        plt.close(fig3)

    # Спектр
    st.subheader("📊 Спектральный анализ")
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
    st.header("📋 Параметры модели")

    model_data = []
    for i in range(len(depths) + 1):
        model_data.append({
            "Слой": i,
            "V (м/с)": velocities[i],
            "ρ (кг/м³)": densities[i],
            "Z (ρ·V)": velocities[i] * densities[i]
        })
    st.table(model_data)

    boundaries_data = []
    for i in range(len(depths)):
        if rc_model.two_way_times[i] < time_axis[-1]:
            boundaries_data.append({
                "Горизонт": i + 1,
                "Глубина (м)": depths[i],
                "Время (с)": f"{rc_model.two_way_times[i]:.3f}",
                "R": f"{rc_model.reflection_coeffs[i]:.4f}"
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

except Exception as e:
    st.error(f"Ошибка: {e}")
    st.markdown("Проверьте корректность введённых параметров (глубины должны возрастать).")