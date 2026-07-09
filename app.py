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
st.title("Генератор синтетических сейсмических трасс")
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

    def generate_from_model(self, depths, velocities, densities, total_time, trace_length=None):
        rc = ReflectionCoefficients(depths, velocities, densities)
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
# 3. ГИБРИДНЫЙ ГЕНЕРАТОР (ИСПРАВЛЕННЫЙ)
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

    def _compute_reflectivity_with_trends(self, depths, velocities, densities, layer_trends):
        """Расчёт коэффициентов отражения с учётом микрослоистости внутри КАЖДОГО слоя."""
        reflectivity = np.zeros(self.num_samples)

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
                t = 2 * (z_micro[j + 1] - top) / avg_v + 2 * top / V0
                idx = int(round(t / self.dt))
                if idx < self.num_samples:
                    reflectivity[idx] += R

        return reflectivity

    def generate_trace(self, distributions, layer_trends,
                       use_mc=True, use_trends=True,
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
                depths, velocities, densities, layer_trends
            )
        else:
            rc = ReflectionCoefficients(depths, velocities, densities)
            reflectivity = rc.get_reflectivity_series(self.dt, self.total_time)

        # Используем trace_gen с правильной амплитудой
        trace = self.trace_gen.generate_from_reflectivity(reflectivity)

        # Обрезаем до нужной длины
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
trace_length = st.sidebar.slider("Длина трассы (количество отсчетов)", 100, 10000, 2000)

# Количество горизонтов
num_horizons = st.sidebar.slider("Количество отражающих горизонтов", 1, 20, 3)

# Параметры дискретизации
dt = st.sidebar.number_input("Интервал дискретизации dt (мс)",
                             min_value=1.0, max_value=4.0, value=1.0) / 1000

# Длительность записи (до 5 секунд)
total_time = st.sidebar.number_input("Длительность записи (с)",
                                     min_value=0.5, max_value=5.0, value=5.0)

# Параметры вейвлета
wavelet_freq = st.sidebar.slider("Частота вейвлета (Гц)", 5, 500, 30)

# Уровень шума
noise_std = st.sidebar.slider("Уровень шума", 0.0, 1.0, 0.05)

# Амплитуда
amplitude = st.sidebar.slider("Амплитуда", 50, 2500, 1000)

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
    min_value=500, max_value=6000, value=default_velocities[0]
))
densities.append(st.sidebar.number_input(
    "Плотность слоя 0 (кг/м³)",
    min_value=1500, max_value=3500, value=default_densities[0]
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
    # Создаём гибридный генератор с амплитудой
    hybrid_gen = HybridGenerator(
        wavelet_freq=wavelet_freq,
        dt=dt,
        total_time=total_time,
        noise_level=noise_std,
        amplitude=amplitude  # Передаём амплитуду
    )

    # Создаём распределения для Монте-Карло
    distributions = {
        'velocities': [{'type': 'normal', 'mu': v, 'sigma': v * 0.05} for v in velocities],
        'densities': [{'type': 'normal', 'mu': r, 'sigma': r * 0.03} for r in densities],
        'depths': [{'type': 'uniform', 'low': d * 0.9, 'high': d * 1.1} for d in depths]
    }

    # Генерируем трассу через гибридный генератор
    result = hybrid_gen.generate_trace(
        distributions, layer_trends,
        use_mc=use_mc,
        use_trends=use_trends,
        seed=42
    )

    trace = result['trace']
    time_axis = result['time_axis']
    mode = result['mode']

    # ✅ НЕ обрезаем трассу, а используем как есть
    # Если нужно изменить длину - делаем интерполяцию или ресэмплинг

    st.header("Результат генерации")
    st.info(f"Режим: **{mode}** | Длина трассы: {len(trace)} отсчетов | Время записи: {time_axis[-1]:.2f} с")

    # Основной график
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, trace, 'b-', linewidth=1.5, label='Синтетическая трасса')

    # Отмечаем горизонты
    rc = ReflectionCoefficients(result['depths'], result['velocities'], result['densities'])
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

except Exception as e:
    st.error(f"Ошибка: {e}")
    st.markdown("Проверьте корректность введённых параметров (глубины должны возрастать).")