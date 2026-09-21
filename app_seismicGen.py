import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert
import io
import segyio


#===============================================================================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def render_interactive_segy_viewer(sgy_buffer, dt_ms):
    """
    Интерактивный просмотрщик SEG-Y файлов.
    Параметры:
    ----------
    sgy_buffer : io.BytesIO        буфер с SEG-Y данными
    dt_ms : float                  интервал дискретизации в миллисекундах
    """
    st.markdown("---")
    st.subheader("Интерактивный просмотрщик SEG-Y")
    
    # Сохранение буфера во временный файл для segyio
    with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        tmp_file.write(sgy_buffer.getvalue())
        
    try:
        with segyio.open(tmp_filename, ignore_geometry=True) as f:
            # Чтение всех данных
            traces = f.trace.raw[:]
            num_traces, num_samples = traces.shape
            
            # Ось времени
            dt_s = dt_ms / 1000.0
            time_axis = np.arange(num_samples) * dt_s
            
            # Нормализация для отображения
            max_val = np.percentile(np.abs(traces), 98)
            if max_val == 0:
                max_val = 1
            traces_norm = traces / max_val
            
            # Вкладки для разных режимов просмотра
            tab1, tab2, tab3, tab4 = st.tabs([
                "Сейсмический разрез", 
                "Wiggle Trace",
                "Variable Area Display",
                "Заголовки трасс"
            ])
            
            # WIGGLE TRACE
            with tab1:
                st.markdown("### Wiggle Trace (Вейвлетная запись)")
                st.info("Классическое отображение сейсмических трасс в виде вейвлетов")
                
                col_w1, col_w2, col_w3 = st.columns([1, 1, 1])
                with col_w1:
                    start_tr = st.number_input(
                        "Начальная трасса", 
                        0, num_traces-1, 0,
                        key="start_tr_wiggle"
                    )
                with col_w2:
                    max_traces = min(num_traces, 200)
                    end_tr = st.number_input(
                        "Конечная трасса", 
                        start_tr+1, num_traces, 
                        min(start_tr+50, num_traces),
                        key="end_tr_wiggle"
                    )
                with col_w3:
                    gain_wiggle = st.slider(
                        "Усиление", 
                        0.1, 10.0, 1.0, 0.1,
                        key="gain_wiggle"
                    )
                
                # Создание фигуры для wiggle
                fig_wiggle = go.Figure()
                
                trace_indices = np.arange(start_tr, min(end_tr, num_traces))
                spacing = 1.0
                
                for i, tr_idx in enumerate(trace_indices):
                    trace_data = traces_norm[tr_idx] * gain_wiggle
                    x_offset = i * spacing
                    
                    fig_wiggle.add_trace(go.Scatter(
                        x=x_offset + trace_data,
                        y=time_axis,
                        mode='lines',
                        line=dict(color='black', width=0.8),
                        name=f'Трасса {tr_idx}',
                        hovertemplate=(
                            f'<b>Трасса:</b> {tr_idx}<br>'
                            '<b>Время:</b> %{y:.4f} с<br>'
                            '<b>Амплитуда:</b> %{x:.4f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ))
                
                fig_wiggle.update_layout(
                    title=f"Wiggle Trace (Трассы {start_tr} - {end_tr-1})",
                    xaxis_title="Номер трассы (со смещением)",
                    yaxis_title="Время (с)",
                    height=700,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=False, zeroline=False),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_wiggle, use_container_width=True)

            # СЕЙСМИЧЕСКИЙ РАЗРЕЗ (HEATMAP)
            with tab2:
                st.markdown("### Сейсмический разрез (Heatmap)")
                
                col_g1, col_g2, col_g3 = st.columns([1, 1, 2])
                with col_g1:
                    gain_heat = st.slider("Усиление (Gain)", 0.1, 10.0, 1.0, 0.1, key="gain_heat")
                with col_g2:
                    cmap = st.selectbox(
                        "Цветовая схема", 
                        ['balance', 'RdBu', 'RdBu_r', 'gray', 'viridis', 'plasma', 'coolwarm'],
                        index=0,  # По умолчанию 'balance'
                        key="cmap_heat"
                    )
                with col_g3:
                    show_contours = st.checkbox("Показать контуры", value=False)

                # Подготовка данных для heatmap
                z_data = np.clip(traces_norm * gain_heat, -1, 1).T

                fig_heat = go.Figure(data=go.Heatmap(
                    z=z_data,
                    x=np.arange(num_traces),
                    y=time_axis,
                    colorscale=cmap,  
                    zmin=-1, 
                    zmax=1,
                    hovertemplate=(
                        '<b>Трасса:</b> %{x}<br>'
                        '<b>Время:</b> %{y:.4f} с<br>'
                        '<b>Амплитуда:</b> %{z:.4f}<br>'
                        '<extra></extra>'
                    ),
                    showscale=True,
                    colorbar=dict(
                        title="Норм. амплитуда",
                        thickness=20,
                        len=0.8
                    )
                ))
                
                # Добавление контуров если нужно
                if show_contours:
                    fig_heat.add_trace(go.Contour(
                        z=z_data,
                        x=np.arange(num_traces),
                        y=time_axis,
                        contours=dict(
                            coloring='none',
                            showlabels=True,
                            labelfont=dict(size=10, color='white')
                        ),
                        line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
                        showscale=False,
                        hoverinfo='skip'
                    ))
                
                fig_heat.update_layout(
                    title=f"Сейсмический разрез ({num_traces} трасс × {num_samples} отсчетов)",
                    xaxis_title="Номер трассы (CDP)",
                    yaxis_title="Время (с)",
                    height=700,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=False),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # Кнопка экспорта
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    if st.button("Экспорт в PNG", key="export_heat_png"):
                        st.info("Используйте панель инструментов Plotly для сохранения изображения")
                with col_exp2:
                    if st.button("Статистика разреза", key="stats_heat"):
                        st.write(f"**Количество трасс:** {num_traces}")
                        st.write(f"**Отсчетов на трассу:** {num_samples}")
                        st.write(f"**Длительность записи:** {time_axis[-1]:.3f} с")
                        st.write(f"**Макс. амплитуда (до нормировки):** {max_val:.2f}")
                        st.write(f"**Средняя амплитуда:** {np.mean(np.abs(traces)):.2f}")
                        st.write(f"**RMS амплитуда:** {np.sqrt(np.mean(traces**2)):.2f}")
                        
            #  VARIABLE AREA DISPLAY (VAD)
            with tab3:
                st.markdown("### Variable Area Display (Закраска фаз)")
                st.info("Закраска положительной (красная) и отрицательной (синяя) фаз")
                
                col_v1, col_v2, col_v3 = st.columns([1, 1, 1])
                with col_v1:
                    start_tr_vad = st.number_input(
                        "Начальная трасса", 
                        0, num_traces-1, 0,
                        key="start_tr_vad"
                    )
                with col_v2:
                    end_tr_vad = st.number_input(
                        "Конечная трасса", 
                        start_tr_vad+1, num_traces,
                        min(start_tr_vad+50, num_traces),
                        key="end_tr_vad"
                    )
                with col_v3:
                    gain_vad = st.slider(
                        "Усиление", 
                        0.1, 10.0, 1.0, 0.1,
                        key="gain_vad"
                    )
                
                # Создание фигуры для VAD
                fig_vad = go.Figure()
                
                trace_indices_vad = np.arange(start_tr_vad, min(end_tr_vad, num_traces))
                
                for i, tr_idx in enumerate(trace_indices_vad):
                    trace_data = traces_norm[tr_idx] * gain_vad
                    x_offset = i * spacing
                    
                    # Положительная фаза (красная)
                    fig_vad.add_trace(go.Scatter(
                        x=x_offset + trace_data,
                        y=time_axis,
                        fill='tozerox',
                        fillcolor='rgba(255, 0, 0, 0.4)',
                        line=dict(color='black', width=0.5),
                        hovertemplate=(
                            f'<b>Трасса:</b> {tr_idx}<br>'
                            '<b>Время:</b> %{y:.4f} с<br>'
                            '<b>Амплитуда:</b> %{x:.4f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ))
                    
                    # Отрицательная фаза (синяя)
                    fig_vad.add_trace(go.Scatter(
                        x=x_offset - trace_data,
                        y=time_axis,
                        fill='tozerox',
                        fillcolor='rgba(0, 0, 255, 0.4)',
                        line=dict(color='black', width=0.5),
                        hovertemplate=(
                            f'<b>Трасса:</b> {tr_idx}<br>'
                            '<b>Время:</b> %{y:.4f} с<br>'
                            '<b>Амплитуда:</b> %{x:.4f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ))
                
                fig_vad.update_layout(
                    title=f"Variable Area Display (Трассы {start_tr_vad} - {end_tr_vad-1})",
                    xaxis_title="Номер трассы (со смещением)",
                    yaxis_title="Время (с)",
                    height=700,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=False, zeroline=False),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_vad, use_container_width=True)
                
                # Легенда
                st.markdown("""
                <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                <b>Легенда:</b><br>
                🔴 <span style='color: red;'>Красная закраска</span> - положительная фаза (пик)<br>
                🔵 <span style='color: blue;'>Синяя закраска</span> - отрицательная фаза (впадина)
                </div>
                """, unsafe_allow_html=True)
            
            # ЗАГОЛОВКИ ТРАСС
            with tab4:
                st.markdown("### Заголовки трасс (Trace Headers)")
                st.info("Просмотр метаданных SEG-Y файла")
                
                selected_tr = st.number_input(
                    "Выберите трассу для просмотра заголовка", 
                    0, num_traces-1, 0,
                    key="selected_tr_header"
                )
                
                # Чтение заголовков
                header = f.header[selected_tr]
                
                # Основные поля заголовка
                header_fields = [
                    (segyio.TraceField.TraceNumber, "Номер трассы"),
                    (segyio.TraceField.CDP, "CDP"),
                    (segyio.TraceField.CDP_TRACE, "CDP Trace"),
                    (segyio.TraceField.TRACE_SEQUENCE_LINE, "Последовательность (линия)"),
                    (segyio.TraceField.TRACE_SEQUENCE_FILE, "Последовательность (файл)"),
                    (segyio.TraceField.SourceX, "X источника"),
                    (segyio.TraceField.SourceY, "Y источника"),
                    (segyio.TraceField.GroupX, "X приемника"),
                    (segyio.TraceField.GroupY, "Y приемника"),
                    (segyio.TraceField.offset, "Offset"),
                    (segyio.TraceField.ReceiverGroupElevation, "Высота приемника"),
                    (segyio.TraceField.SourceSurfaceElevation, "Высота источника"),
                ]
                
                header_data = []
                for field, description in header_fields:
                    try:
                        value = header[field]
                        header_data.append({
                            "Поле": description,
                            "Код": field.name,
                            "Значение": value
                        })
                    except:
                        pass
                
                st.dataframe(
                    pd.DataFrame(header_data),
                    use_container_width=True,
                    height=400
                )
                
                # Бинарный заголовок файла
                st.markdown("### Бинарный заголовок файла")
                bin_header_data = []
                bin_fields = [
                    (segyio.BinField.Interval, "Интервал дискретизации (мкс)"),
                    (segyio.BinField.Samples, "Количество отсчетов"),
                    (segyio.BinField.Format, "Формат данных"),
                ]
                
                for field, description in bin_fields:
                    try:
                        value = f.bin[field]
                        bin_header_data.append({
                            "Поле": description,
                            "Код": field.name,
                            "Значение": value
                        })
                    except:
                        pass
                
                st.dataframe(
                    pd.DataFrame(bin_header_data),
                    use_container_width=True
                )
                
                # Текстовый заголовок
                st.markdown("### Текстовый заголовок")
                try:
                    text_header = f.text[0].decode('ascii', errors='ignore')
                    st.code(text_header, language='text')
                except:
                    st.warning("Не удалось прочитать текстовый заголовок")
    
    finally:
        # Удаление временного файл
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)
#============================================================================================================================================


st.set_page_config(page_title="Генератор сейсмических трасс", layout="wide")
st.title("Генератор синтетических сейсмических трасс")
st.markdown("Настройте параметры геологической модели и получите синтетическую трассу в реальном времени.")




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






# КЛАССЫ ДЛЯ ГЕНЕРАЦИИ

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


# ГИБРИДНЫЙ ГЕНЕРАТОР

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
    

    def generate_section_with_lateral_variations(self, depths_base, velocities, densities, 
                                             horizon_index=0, variation_amplitude=50, 
                                             variation_type='sinusoidal', num_traces=100,
                                             use_mc=False, use_trends=False,
                                             distributions=None, layer_trends=None):
        """
        Генерация синтетического разреза с латеральными изменениями.
        """
        num_samples = self.num_samples
        time_axis = self.time_axis
        
        # Массив глубин для каждой трассы
        x_positions = np.arange(num_traces)
        
        # Генерация вариаций глубины
        if variation_type == 'sinusoidal':
            variation = variation_amplitude * np.sin(2 * np.pi * x_positions / num_traces)
        elif variation_type == 'linear':
            variation = variation_amplitude * (2 * x_positions / num_traces - 1)
        elif variation_type == 'random':
            variation = np.random.randn(num_traces) * variation_amplitude / 2
            from scipy.ndimage import uniform_filter1d
            variation = uniform_filter1d(variation, size=5)
        else:
            variation = np.zeros(num_traces)
        
        # Глубины для каждой трассы
        depths_array = []
        traces = []
        
        for i in range(num_traces):
            # Копия базовых глубин
            depths_i = depths_base.copy()
            # Изменение выбранного горизонта
            depths_i[horizon_index] = depths_base[horizon_index] + variation[i]
            depths_array.append(depths_i[horizon_index])
            
            # Генерация трассы с использованием гибридного подхода
            if use_mc and distributions is not None:
                # Монте-Карло: случайные параметры для каждой трассы
                mc_depths, mc_velocities, mc_densities = self.sample_geology(distributions)
                # Корректируем глубину с учетом латеральных изменений
                mc_depths[horizon_index] = mc_depths[horizon_index] + variation[i]
                
                if use_trends and layer_trends is not None:
                    reflectivity = self._compute_reflectivity_with_trends(
                        mc_depths, mc_velocities, mc_densities, layer_trends
                    )
                else:
                    rc = ReflectionCoefficients(mc_depths, mc_velocities, mc_densities)
                    reflectivity = rc.get_reflectivity_series(self.dt, self.total_time)
            else:
                # Детерминированный режим с латеральными изменениями
                if use_trends and layer_trends is not None:
                    reflectivity = self._compute_reflectivity_with_trends(
                        depths_i, velocities, densities, layer_trends
                    )
                else:
                    rc = ReflectionCoefficients(depths_i, velocities, densities)
                    reflectivity = rc.get_reflectivity_series(self.dt, self.total_time)
            
            # Генерация трассы
            trace = self.trace_gen.generate_from_reflectivity(reflectivity)
            
            # Обрезаем до нужной длины
            if len(trace) > num_samples:
                trace = trace[:num_samples]
            elif len(trace) < num_samples:
                pad_len = num_samples - len(trace)
                trace = np.pad(trace, (0, pad_len), 'constant')
            
            traces.append(trace)
        
        traces = np.array(traces)
        depths_array = np.array(depths_array)
        
        return traces, time_axis, depths_array


#  БОКОВАЯ ПАНЕЛЬ С ПАРАМЕТРАМИ
st.sidebar.header("⚙️ Параметры модели")

# ЧЕКБОКСЫ ДЛЯ ГИБРИДНОГО ГЕНЕРАТОРА
st.sidebar.subheader("Режимы генерации")
use_mc = st.sidebar.checkbox("Учитывать неопределённость (Монте-Карло)", value=False)
use_trends = st.sidebar.checkbox("Учитывать неоднородность слоёв", value=False)
enable_lateral = st.sidebar.checkbox("Включить латеральные изменения", value=False)

# Длина трассы
st.sidebar.subheader("Параметры трассы")

# Количество горизонтов
num_horizons = st.sidebar.slider("Количество отражающих горизонтов", 1, 20, 3)

# Параметры дискретизации
dt = st.sidebar.number_input("Интервал дискретизации dt (мс)",
                             min_value=0.5, max_value=4.0, value=1.0) / 1000

# Длительность записи (до 5 секунд)
total_time = st.sidebar.number_input("Длительность записи (с)",
                                     min_value=0.5, max_value=5.0, value=5.0)


# Латеральные изменения
if enable_lateral:
    horizon_to_vary = st.sidebar.selectbox(
        "Горизонт для изменения",
        options=list(range(1, num_horizons + 1)),
        format_func=lambda x: f"Горизонт {x}",
        index=0
    )
    
    variation_type = st.sidebar.selectbox(
        "Тип изменения",
        ['sinusoidal', 'linear', 'random'],
        index=0
    )
    
    variation_amplitude = st.sidebar.slider(
        "Амплитуда изменения (м)",
        min_value=0,
        max_value=500,
        value=50,
        step=10
    )

# Параметры вейвлета
wavelet_freq = st.sidebar.slider("Частота вейвлета (Гц)", 5, 500, 30)

# Уровень шума
noise_std = st.sidebar.slider("Уровень шума", 0.0, 1.0, 0.0)

# Амплитуда
amplitude = st.sidebar.slider("Амплитуда", 50, 2500, 1000)

# Количество трасс
num_traces_to_generate = st.sidebar.number_input(
    "Количество трасс для генерации",
    min_value=1,
    max_value=50000,
    value=100,
    step=1
) #help="Количество синтетических трасс для генерации (для вертикального отображения)"

# ГЕОЛОГИЧЕСКАЯ МОДЕЛЬ
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
        f"Глубина границы {i + 1} (м)",
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
    if use_trends:
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
    else:
        # Если микрослоистость выключена, добавляем пустые тренды
        layer_trends.append({'type': 'none'})

# ГЕНЕРАЦИЯ И ОТОБРАЖЕНИЕ
try:
    # Гибридный генератор с амплитудой
    hybrid_gen = HybridGenerator(
        wavelet_freq=wavelet_freq,
        dt=dt,
        total_time=total_time,
        noise_level=noise_std,
        amplitude=amplitude  # Передаём амплитуду
    )

    # Распределения для Монте-Карло
    distributions = {
        'velocities': [{'type': 'normal', 'mu': v, 'sigma': v * 0.05} for v in velocities],
        'densities': [{'type': 'normal', 'mu': r, 'sigma': r * 0.03} for r in densities],
        'depths': [{'type': 'uniform', 'low': d * 0.9, 'high': d * 1.1} for d in depths]
    }

    # Трассу через гибридный генератор
    result = hybrid_gen.generate_trace(
        distributions, layer_trends,
        use_mc=use_mc,
        use_trends=use_trends,
        seed=42
    )

    trace = result['trace']
    time_axis = result['time_axis']
    mode = result['mode']

    # НЕ обрезаем трассу, а используем как есть
    # Если нужно изменить длину - делаем интерполяцию или ресэмплинг

    st.header("Результат генерации")
    st.info(f"Режим: **{mode}** | Длина трассы: {len(trace)} отсчетов | Время записи: {time_axis[-1]:.2f} с")

    # Основной график
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, trace, 'b-', linewidth=1.5, label='Синтетическая трасса')

    # Отметки горизонтов
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

    if num_traces_to_generate > 1:
        st.subheader("Ансамбль синтетических трасс")
        
        with st.spinner(f"Генерация {num_traces_to_generate} трасс..."):
            traces_ensemble = []
            
            if enable_lateral:
                # Режим с латеральными изменениями
                horizon_idx = horizon_to_vary - 1  # Переводим в 0-индексацию
                
                traces_ensemble, time_axis_ensemble, depths_varied = hybrid_gen.generate_section_with_lateral_variations(
                    depths_base=result['depths'],
                    velocities=result['velocities'],
                    densities=result['densities'],
                    horizon_index=horizon_idx,
                    variation_amplitude=variation_amplitude,
                    variation_type=variation_type,
                    num_traces=num_traces_to_generate,
                    use_mc=use_mc,
                    use_trends=use_trends,
                    distributions=distributions if use_mc else None,
                    layer_trends=layer_trends if use_trends else None
                )
                
                # Информация о генерации
                st.info(f"Латеральные изменения: {variation_type}, амплитуда {variation_amplitude} м, горизонт {horizon_to_vary}")
            else:
                # Параметры из уже сгенерированной трассы
                base_depths = result['depths']
                base_velocities = result['velocities']
                base_densities = result['densities']
                
                # Генерируем множество трасс
                for i in range(num_traces_to_generate):
                    if use_mc:
                        # Монте-Карло: каждая трасса со случайными параметрами
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
                        
                        # Генерация трассы со случайным seed
                        result_i = hybrid_gen.generate_trace(
                            distributions, layer_trends_i,
                            use_mc=use_mc,
                            use_trends=use_trends,
                            seed=None    # Установить seed для Монте-Карло
                        )
                    else:
                        # Детерминированный режим: ВСЕ ТРАССЫ ИДЕНТИЧНЫЕ. Проблема - нет режима вариации без Монте-Карло как такового
                        result_i = hybrid_gen.generate_trace(
                            distributions, layer_trends,
                            use_mc=use_mc,
                            use_trends=use_trends,
                            seed=42  # Фиксированный seed
                        )
                    
                    traces_ensemble.append(result_i['trace'])
                
                traces_ensemble = np.array(traces_ensemble)

        
        # Вертикальный график ансамбля (как в SEG-Y)
        fig_ensemble, ax_ensemble = plt.subplots(figsize=(12, 10))
        
        # Масштаб для отображения
        max_amplitude = np.max(np.abs(traces_ensemble))
        if max_amplitude > 0:
            # Нормализуем трассы
            normalized_traces = traces_ensemble / max_amplitude
            
            # Трассы вертикально (как в сейсмических разрезах)
            trace_spacing = 1.0
            num_show = min(len(traces_ensemble), 200)
            
            # Режим для подписи
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
            
            # Ось Y (время возрастает вниз, как в сейсмике)
            ax_ensemble.invert_yaxis()
            
            # Настройка осей
            ax_ensemble.set_xlim(-1.5, num_show * trace_spacing + 1.5)
            ax_ensemble.set_ylim(time_axis[-1], 0)  # Время сверху вниз
            
            # Отметки горизонтальных линий (горизонты)
            # Горизонты теперь будут горизонтальными линиями на определенных временах
            rc_ensemble = ReflectionCoefficients(result['depths'], result['velocities'], result['densities'])
            for t in rc_ensemble.two_way_times:
                if t < time_axis[-1]:
                    ax_ensemble.axhline(y=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Цветовая шкала или подписи
            # Номера трасс на оси X
            if num_show <= 50:
                x_ticks = np.arange(0, num_show * trace_spacing, trace_spacing * max(1, num_show // 10))
                x_labels = [f"{int(i)}" for i in x_ticks / trace_spacing]
                ax_ensemble.set_xticks(x_ticks)
                ax_ensemble.set_xticklabels(x_labels)
            
            # # Подписи к горизонтам
            # for i, t in enumerate(rc_ensemble.two_way_times):
            #     if t < time_axis[-1]:
            #         ax_ensemble.text(
            #             num_show * trace_spacing * 0.02, 
            #             t, 
            #             f' Г{rc_ensemble.reflection_coeffs[i]:.3f}', 
            #             color='red', 
            #             fontsize=8,
            #             va='center'
            #         )
            
            st.pyplot(fig_ensemble)
            plt.close(fig_ensemble)
            


    plt.close(fig)

    # Включены латеральные изменения - график вариаций глубин
    if enable_lateral and 'depths_varied' in locals():
        st.subheader("Вариации глубин горизонта")
        fig_var, ax_var = plt.subplots(figsize=(12, 4))
        ax_var.plot(np.arange(len(depths_varied)), depths_varied, 'b-', linewidth=2)
        ax_var.axhline(y=result['depths'][horizon_idx], color='r', linestyle='--', 
                    label=f'Базовое значение: {result["depths"][horizon_idx]:.1f} м')
        ax_var.set_xlabel('Номер трассы')
        ax_var.set_ylabel('Глубина (м)')
        ax_var.set_title(f'Изменение глубины горизонта {horizon_to_vary} ({variation_type})')
        ax_var.grid(True, alpha=0.3)
        ax_var.legend()
        st.pyplot(fig_var)
        plt.close(fig_var)

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
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Save ансамбль трасс в SEG-Y
            save_traces_to_sgy(
                traces=traces_ensemble,
                dt_ms=dt * 1000,
                filename=tmp_filename,
                title=f"Синтетический разрез ({mode})"
            )
            
            # Read файл в буфер
            with open(tmp_filename, 'rb') as f:
                sgy_buffer = io.BytesIO(f.read())
            
            st.download_button(
                label=f"Скачать все трассы SEG-Y",
                data=sgy_buffer.getvalue(),
                file_name="generated_traces.sgy",
                mime="application/octet-stream"
            )
        finally:
            # К чертям временный файл
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

        

    # ЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭЭ
    if 'sgy_buffer' in locals() and sgy_buffer is not None:
        render_interactive_segy_viewer(sgy_buffer, dt_ms=dt * 1000)



except Exception as e:
    st.error(f"Ошибка: {e}")
    st.markdown("Проверьте корректность введённых параметров (глубины должны возрастать).")