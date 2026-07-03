def well_to_synthetic(depths, v, rho, wavelet_freq=30, dt=0.001, total_time=None):
    """
    Преобразует скважинные данные (глубины, скорости, плотности) в синтетическую трассу.
    """
    if total_time is None:
        max_time = 2 * np.max(depths) / np.min(v[v>0])  # оценка максимального времени
        total_time = max_time + 0.5
    
    # Шаг 1: выделяем границы слоёв и средние значения
    # Для простоты: берём резкие изменения импеданса
    z = v * rho
    dZ = np.diff(z)
    
    # Находим границы, где изменение импеданса значительное (>5% от среднего)
    boundaries = np.where(np.abs(dZ) > 0.05 * np.mean(z))[0]
    
    depths_boundary = depths[boundaries + 1]  # глубины границ
    # Коэффициенты отражения (приближённо)
    R_boundary = dZ[boundaries] / (z[boundaries] + z[boundaries + 1])
    
    # Шаг 2: средние скорости и плотности между границами
    # ... (упрощённо для демонстрации)
    
    # Генерируем трассу
    rc = ReflectionCoefficients(depths_boundary, 
                                 v[boundaries[:len(depths_boundary)]+1], 
                                 rho[boundaries[:len(depths_boundary)]+1])
    reflectivity = rc.get_reflectivity_series(dt, total_time)
    generator = TraceGenerator(wavelet_freq, dt)
    trace = generator.generate(reflectivity)
    
    return trace, rc

# Пример генерации синтетики по псевдо-каротажу
trace_from_well, rc_well = well_to_synthetic(depths_well, v_well, rho_well)

time_well = np.arange(0, len(trace_from_well) * dt, dt)
plt.plot(time_well, trace_from_well)
plt.title('Синтетическая трасса по скважинным данным')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()