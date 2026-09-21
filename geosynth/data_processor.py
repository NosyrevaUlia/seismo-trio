import lasio
import pandas as pd
import numpy as np
import os

def get_processed_data(csv_path, facies_target=2, n_points=200):
    intervals = pd.read_csv(csv_path, encoding='utf-8-sig')
    gk_list, ps_list, depths_list, well_names = [], [], [], []
    
    target_intervals = intervals[intervals['фация'] == facies_target]
    
    for _, row in target_intervals.iterrows():
        well_id = int(row['скважина'])
        well_file = f"{well_id}.las"
        if not os.path.exists(well_file): continue
            
        try:
            las = lasio.read(well_file, encoding='cp1251')
            df = las.df().reset_index()
            
            curves = [c.mnemonic for c in las.curves if c.mnemonic != 'DEPT']
            if len(curves) < 2: continue
            gk_col, ps_col = curves[0], curves[1]
            
            interval = df[(df['DEPT'] >= row['кровля']) & (df['DEPT'] <= row['подошва'])].copy()
            interval = interval.dropna(subset=['DEPT', gk_col, ps_col])
            if len(interval) < 2: continue
            
            target_depths = np.linspace(row['кровля'], row['подошва'], n_points)
            
            gk_vals = np.interp(target_depths, interval['DEPT'], interval[gk_col])
            ps_vals = np.interp(target_depths, interval['DEPT'], interval[ps_col])
            
            gk_list.append(gk_vals)
            ps_list.append(ps_vals)
            depths_list.append(target_depths)
            well_names.append(f"Скв. {well_id}")
            
        except Exception as e:
            print(f"Ошибка в {well_file}: {e}")
            
    X_gk = np.array(gk_list)
    X_ps = np.array(ps_list)
    X_depths = np.array(depths_list)
    
    if X_gk.size == 0:
        raise ValueError("Данные не загружены! Проверьте файлы в папке.")
        
    mean_gk, std_gk = np.nanmean(X_gk), np.nanstd(X_gk)
    mean_ps, std_ps = np.nanmean(X_ps), np.nanstd(X_ps)
    
    norm_gk = np.nan_to_num((X_gk - mean_gk) / (std_gk if std_gk != 0 else 1.0))
    norm_ps = np.nan_to_num((X_ps - mean_ps) / (std_ps if std_ps != 0 else 1.0))
    
    X_norm = np.stack([norm_gk, norm_ps], axis=1)
    
    return X_norm, (mean_gk, std_gk, mean_ps, std_ps), X_gk, X_ps, X_depths, well_names


