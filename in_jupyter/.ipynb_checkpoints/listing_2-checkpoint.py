import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1. Класс для вейвлета Рикера (у Юли уже есть, но включу для целостности)
##############################################################################

class RickerWavelet:
    """Генератор вейвлета Рикера с заданными параметрами."""
    
    def __init__(self, dominant_freq=30, dt=0.001, length=0.2):
        """
        dominant_freq : доминантная частота (Гц)
        dt            : интервал дискретизации (с)
        length        : длительность вейвлета (с)
        """
        self.freq = dominant_freq
        self.dt = dt
        self.length = length
        
        # Временная сетка для вейвлета (центрируем в середине окна)
        self.time = np.arange(-length/2, length/2, dt)
        self.wavelet = self._ricker()
    
    def _ricker(self):
        """Вычисление вейвлета по формуле."""
        t = self.time
        f = self.freq
        tau = np.pi * f * t
        wavelet = (1 - 2 * tau**2) * np.exp(-tau**2)
        return wavelet
    
    def get_wavelet(self):
        """Возвращает временную сетку и вейвлет."""
        return self.time, self.wavelet
    
    def plot(self):
        """Визуализация вейвлета."""
        plt.figure(figsize=(8, 3))
        plt.plot(self.time, self.wavelet)
        plt.title(f'Вейвлет Рикера, f = {self.freq} Гц')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.grid(True)
        plt.show()


##############################################################################
# 2. Класс для расчёта коэффициентов отражения по импедансам
##############################################################################

class ReflectionCoefficients:
    """
    Расчёт коэффициентов отражения для горизонтально-слоистой среды.
    """
    
    def __init__(self, depths, velocities, densities):
        """
        depths     : массив глубин границ (м) от поверхности
                     Например, [500, 1200, 2000] 
                     (первая граница на 500 м, вторая на 1200 м...)
        velocities : массив скоростей в слоях (м/с)
                     Длина = len(depths) + 1 (для каждого слоя)
        densities  : массив плотностей (кг/м³) 
                     Длина = len(depths) + 1
        """
        self.depths = np.array(depths)
        self.velocities = np.array(velocities)
        self.densities = np.array(densities)
        self.num_layers = len(velocities)
        self.num_boundaries = len(depths)
        
        # Рассчитаем импедансы
        self.impedances = self._compute_impedances()
        
        # Рассчитаем коэффициенты отражения
        self.reflection_coeffs = self._compute_reflection_coeffs()
        
        # Рассчитаем времена прихода (двустороннее время)
        self.two_way_times = self._compute_times()
    
    def _compute_impedances(self):
        """Акустический импеданс Z = ρ * V."""
        return self.densities * self.velocities
    
    def _compute_reflection_coeffs(self):
        """
        Коэффициент отражения на границе между слоями i и i+1:
        R = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i)
        """
        R = []
        for i in range(self.num_boundaries):
            Z_upper = self.impedances[i]
            Z_lower = self.impedances[i+1]
            coeff = (Z_lower - Z_upper) / (Z_lower + Z_upper)
            R.append(coeff)
        return np.array(R)
    
    def _compute_times(self):
        """
        Двустороннее время прихода отражённой волны от каждой границы.
        Для границы на глубине H: t = sum(2 * толщина_слоя / скорость_слоя)
        """
        times = []
        cumulative_time = 0.0
        
        # Времена для каждой границы
        for i in range(self.num_boundaries):
            # Толщина i-го слоя (от предыдущей границы до текущей)
            if i == 0:
                thickness = self.depths[0]  # слой 0 от поверхности до первой границы
            else:
                thickness = self.depths[i] - self.depths[i-1]
            
            # Двустороннее время для этого слоя
            layer_time = 2 * thickness / self.velocities[i]
            cumulative_time += layer_time
            times.append(cumulative_time)
        
        return np.array(times)
    
    def get_reflectivity_series(self, dt, total_time):
        """
        Преобразует коэффициенты отражения в дискретную временную серию.
        
        dt         : интервал дискретизации (с)
        total_time : полное время записи (с)
        
        Возвращает:
        time_axis  : массив времени
        reflectivity : массив коэффициентов (палки на временах t_i)
        """
        num_samples = int(total_time / dt) + 1
        time_axis = np.arange(0, total_time, dt)
        reflectivity = np.zeros(num_samples)
        
        for t, r in zip(self.two_way_times, self.reflection_coeffs):
            idx = int(t / dt)
            if idx < num_samples:
                reflectivity[idx] = r
            else:
                print(f"Предупреждение: время {t:.4f} с выходит за окно записи")
        
        return time_axis, reflectivity
    
    def print_model(self):
        """Вывод модели на экран."""
        print("=" * 60)
        print("ГЕОЛОГИЧЕСКАЯ МОДЕЛЬ")
        print("=" * 60)
        print(f"{'Слой':<6} {'V (м/с)':<10} {'ρ (кг/м³)':<12} {'Z (м/с*кг/м³)':<18} {'Толщина (м)':<12}")
        print("-" * 60)
        
        # Первый слой (до первой границы)
        print(f"{'0':<6} {self.velocities[0]:<10.0f} {self.densities[0]:<12.1f} {self.impedances[0]:<18.2f} {self.depths[0]:<12.1f}")
        
        for i in range(1, self.num_layers - 1):
            thickness = self.depths[i] - self.depths[i-1]
            print(f"{i:<6} {self.velocities[i]:<10.0f} {self.densities[i]:<12.1f} {self.impedances[i]:<18.2f} {thickness:<12.1f}")
        
        # Последний слой (бесконечный или до последней границы)
        if self.num_layers > 1:
            thickness = self.depths[-1] - self.depths[-2] if len(self.depths) >= 2 else self.depths[0]
            print(f"{self.num_layers-1:<6} {self.velocities[-1]:<10.0f} {self.densities[-1]:<12.1f} {self.impedances[-1]:<18.2f} infinity")
        
        print("\n" + "=" * 60)
        print("ГРАНИЦЫ")
        print("=" * 60)
        for i in range(self.num_boundaries):
            print(f"Граница {i}/{i+1}: глубина {self.depths[i]} м, R = {self.reflection_coeffs[i]:.4f}, t = {self.two_way_times[i]:.4f} с")


##############################################################################
# 3. Класс для генерации синтетической трассы
##############################################################################

class SeismicTraceGenerator:
    """
    Генератор синтетической сейсмической трассы на основе геологической модели.
    """
    
    def __init__(self, wavelet_freq=30, dt=0.001, noise_std=0.0):
        """
        wavelet_freq : доминантная частота вейвлета (Гц)
        dt           : интервал дискретизации (с)
        noise_std    : стандартное отклонение шума (0 = без шума)
        """
        self.wavelet_freq = wavelet_freq
        self.dt = dt
        self.noise_std = noise_std
        
        # Создаём вейвлет (фиксированной длины для свёртки)
        wavelet_length = 0.2  # длительность вейвлета (с)
        wavelet_time = np.arange(-wavelet_length/2, wavelet_length/2, dt)
        tau = np.pi * wavelet_freq * wavelet_time
        self.wavelet = (1 - 2 * tau**2) * np.exp(-tau**2)
    
    def generate_from_reflectivity(self, reflectivity_series):
        """
        Генерация трассы из временной серии коэффициентов отражения.
        
        reflectivity_series : массив коэффициентов отражения (палки)
        
        Возвращает:
        trace : синтетическая трасса (свёртка + шум)
        """
        # Свёртка
        trace = np.convolve(reflectivity_series, self.wavelet, mode='same')
        
        # Добавление шума
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, len(trace))
            trace += noise
        
        return trace
    
    def generate_from_model(self, depths, velocities, densities, total_time):
        """
        Полная генерация: модель → коэффициенты → свёртка → трасса.
        
        depths      : массив глубин границ (м)
        velocities  : массив скоростей по слоям
        densities   : массив плотностей по слоям
        total_time  : полное время записи (с)
        """
        # Шаг 1: вычисляем коэффициенты отражения
        rc = ReflectionCoefficients(depths, velocities, densities)
        time_axis, reflectivity = rc.get_reflectivity_series(self.dt, total_time)
        
        # Шаг 2: генерируем трассу
        trace = self.generate_from_reflectivity(reflectivity)
        
        return time_axis, trace, rc
    
    def plot_trace(self, time_axis, trace, title="Синтетическая сейсмическая трасса"):
        """Визуализация трассы."""
        plt.figure(figsize=(12, 6))
        
        # Основной график трассы
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, trace, 'b-', linewidth=1.5)
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.title(title)
        plt.grid(True)
        
        # Спектр (опционально, для анализа)
        plt.subplot(1, 2, 2)
        freq = np.fft.rfftfreq(len(trace), d=time_axis[1] - time_axis[0])
        spectrum = np.abs(np.fft.rfft(trace))
        plt.plot(freq, spectrum, 'r-')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда спектра')
        plt.title('Амплитудный спектр трассы')
        plt.grid(True)
        plt.xlim(0, 100)  # ограничим до 100 Гц
        
        plt.tight_layout()
        plt.show()


##############################################################################
# 4. ПРИМЕР ИСПОЛЬЗОВАНИЯ
##############################################################################

if __name__ == "__main__":
    
    # ------------------------------------------------------------------------
    # ПРИМЕР 1. Три слоя, две границы
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("ПРИМЕР 1: Трёхслойная модель (почва → песок → известняк)")
    print("="*60)
    
    # Геологическая модель
    # Глубины границ (от поверхности)
    depths_ex1 = [500, 1500]  # первая граница на 500 м, вторая на 1500 м
    
    # Свойства слоёв (слой 0: выше первой границы, слой 1: между границами, слой 2: ниже второй)
    velocities_ex1 = [1800, 2200, 3500]   # м/с
    densities_ex1  = [2000, 2200, 2500]   # кг/м³
    
    # Параметры генерации
    wavelet_freq = 30      # Гц
    dt = 0.001             # 1 мс
    total_time = 2.0       # 2 секунды записи
    noise_std = 0.01       # небольшой шум
    
    # Создаём генератор
    generator = SeismicTraceGenerator(wavelet_freq=wavelet_freq, dt=dt, noise_std=noise_std)
    
    # Генерируем трассу
    time_axis, trace, rc_model = generator.generate_from_model(depths_ex1, velocities_ex1, densities_ex1, total_time)
    
    # Выводим модель на экран
    rc_model.print_model()
    
    # Визуализируем коэффициенты отражения
    _, reflectivity = rc_model.get_reflectivity_series(dt, total_time)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.stem(time_axis, reflectivity, linefmt='r-', markerfmt='ro', basefmt='k-', use_line_collection=True)
    plt.title('Коэффициенты отражения (палки на временах границ)')
    plt.ylabel('R')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    wavelet_time = np.arange(-0.1, 0.1, dt)
    tau = np.pi * wavelet_freq * wavelet_time
    wavelet_plot = (1 - 2 * tau**2) * np.exp(-tau**2)
    plt.plot(wavelet_time, wavelet_plot)
    plt.title(f'Вейвлет Рикера (f = {wavelet_freq} Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, trace)
    plt.title('Синтетическая трасса (свёртка + шум)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------------
    # ПРИМЕР 2. Модель с отрицательным коэффициентом отражения (смена фазы)
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("ПРИМЕР 2: Отрицательный коэффициент (мягкие породы под жёсткими)")
    print("="*60)
    
    # Жёсткий слой (известняк) сверху, мягкий (песок) снизу
    depths_ex2 = [800]
    velocities_ex2 = [3500, 2000]   # скорость падает
    densities_ex2 = [2600, 2100]    # плотность падает
    
    rc_model2 = ReflectionCoefficients(depths_ex2, velocities_ex2, densities_ex2)
    rc_model2.print_model()
    
    time_axis2, trace2, _ = generator.generate_from_model(depths_ex2, velocities_ex2, densities_ex2, total_time)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis2, trace2)
    plt.title('Трасса с отрицательным коэффициентом отражения (изменение фазы)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()
    
    # ------------------------------------------------------------------------
    # ПРИМЕР 3. Сравнение с одногоризонтной моделью (как у Юли было)
    # ------------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("ПРИМЕР 3: Один горизонт (для сравнения с предыдущей работой)")
    print("="*60)
    
    # Самый простой случай: один горизонт на 1000 м
    depths_ex3 = [1000]
    velocities_ex3 = [2500, 3000]   # небольшое увеличение скорости
    densities_ex3 = [2200, 2400]    # небольшое увеличение плотности
    
    rc_model3 = ReflectionCoefficients(depths_ex3, velocities_ex3, densities_ex3)
    rc_model3.print_model()
    
    generator3 = SeismicTraceGenerator(wavelet_freq=30, dt=0.001, noise_std=0.0)
    time_axis3, trace3, _ = generator3.generate_from_model(depths_ex3, velocities_ex3, densities_ex3, 1.5)
    
    plt.plot(time_axis3, trace3)
    plt.title('Один горизонт (расчёт R через импеданс)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()