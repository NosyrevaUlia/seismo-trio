import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter1d
from data_processor import get_processed_data

# 1. Настройка академического стиля (Times New Roman)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Установка случайных чисел для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# --- ШАГ 1: ЗАГРУЗКА И НОРМАЛИЗАЦИЯ ДАННЫХ ---
print("Загрузка данных из intervals.csv...")
X_norm, params, X_gk, X_ps, X_depths, well_names = get_processed_data('intervals.csv')[:6]
n_points = X_gk.shape[1]
target_depths = X_depths[0]

# Масштабируем данные в диапазон [-1, 1] для стабильности обучения нейросети
gk_min, gk_max = np.min(X_gk), np.max(X_gk)
ps_min, ps_max = np.min(X_ps), np.max(X_ps)

def scale_to_net(val, v_min, v_max):
    return 2.0 * (val - v_min) / (v_max - v_min) - 1.0

def scale_to_phys(val, v_min, v_max):
    return (val + 1.0) / 2.0 * (v_max - v_min) + v_min

X_gk_scaled = scale_to_net(X_gk, gk_min, gk_max)
X_ps_scaled = scale_to_net(X_ps, ps_min, ps_max)

# Объединяем ГК и ПС в один обучающий тензор
train_data = np.stack([X_gk_scaled, X_ps_scaled], axis=1)
train_tensor = torch.FloatTensor(train_data)


# --- ШАГ 2: АРХИТЕКТУРА УСЛОВНОГО АВТОКОДИРОВЩИКА (CVAE) ---
class GeologicalCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(GeologicalCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Энкодер: сжимает каротаж в латентные переменные
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Декодер: восстанавливает ГК и ПС обратно
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim * 2),
            nn.Tanh()  # Выход строго в диапазоне [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon.view(batch_size, 2, self.input_dim), mu, logvar

    def decode(self, z):
        recon = self.decoder(z)
        return recon.view(-1, 2, self.input_dim)


# --- ШАГ 3: ФУНКЦИЯ ПОТЕРЬ С МИНИМАЛЬНЫМ КЛ-ШТРАФОМ ---
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    recon_gk = recon_x[:, 0, :]
    recon_ps = recon_x[:, 1, :]
    corr_penalty = torch.sum(recon_gk * recon_ps)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * corr_penalty + 0.005 * kl_loss


# --- ШАГ 4: ОБУЧЕНИЕ МОДЕЛИ НА ТВОИХ ДАННЫХ ---
print("Обучение CVAE геологическим закономерностям твоего пласта...")
model = GeologicalCVAE(input_dim=n_points, latent_dim=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1500
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    recon, mu, logvar = model(train_tensor)
    loss = loss_function(recon, train_tensor, mu, logvar)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 300 == 0:
        print(f"Эпоха {epoch+1}/{epochs} | Текущая ошибка модели: {loss.item():.4f}")

print("Обучение завершено! Модель зафиксировала скрытые параметры реальных кривых.")


# --- ШАГ 5: СТОХАСТИЧЕСКАЯ ГЕНЕРАЦИЯ ВАРИАНТОВ И СОХРАНЕНИЕ ВЕСОВ ---
print("Генерация 3 реалистичных вариантов...")
model.eval()
n_variants = 3

def generate_synthetic_wells(seed):
    local_rng = np.random.default_rng(seed)
    z_samples = local_rng.normal(0.0, 1.65, size=(n_variants, 4))
    z_tensor = torch.FloatTensor(z_samples)
    
    with torch.no_grad():
        generated = model.decode(z_tensor).numpy()
        
    gk_vars, ps_vars = [], []
    for v in range(n_variants):
        gk_phys = scale_to_phys(generated[v, 0, :], gk_min, gk_max)
        ps_phys = scale_to_phys(generated[v, 1, :], ps_min, ps_max)
        
        gk_final = gaussian_filter1d(gk_phys, sigma=0.8)
        ps_final = gaussian_filter1d(ps_phys, sigma=1.5)
        
        gk_vars.append(gk_final)
        ps_vars.append(ps_final)
        
    return gk_vars, ps_vars

hybrids_gk_g1, hybrids_ps_g1 = generate_synthetic_wells(seed=101)
hybrids_gk_g2, hybrids_ps_g2 = generate_synthetic_wells(seed=202)

# Сохранение весов для Streamlit сразу после генерации
torch.save({
    'model_state': model.state_dict(),
    'input_dim': n_points,
    'gk_min': float(gk_min),
    'gk_max': float(gk_max),
    'ps_min': float(ps_min),
    'ps_max': float(ps_max)
}, "cvae_weights.pt")
print("Файл cvae_weights.pt успешно создан и готов к интеграции!")


# --- ШАГ 6: СТРОГАЯ ОТРИСОВКА ПЛАНШЕТОВ ---
output_dir = "cvae_outputs"
os.makedirs(output_dir, exist_ok=True)

# Динамическое распределение скважин под фактический размер датасета
n_total = len(well_names)
mid = n_total // 2
g1_idx = list(range(0, mid + 1))
g2_idx = list(range(mid + 1, n_total))

all_colors = ['#1f77b4', '#ff770e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
variant_colors = ['#e31a1c', '#33a02c', '#17becf']

def draw_cvae_plate(data_raw, hybrids_g1, hybrids_g2, method_name, unit_name, file_label, x_limits=None):
    fig, axs = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={'wspace': 0.25})
    
    # Левый планшет (Группа 1)
    for c_idx, idx in enumerate(g1_idx):
        axs[0].plot(data_raw[idx], X_depths[idx], color=all_colors[c_idx % len(all_colors)], alpha=0.25, linewidth=1.0, label=well_names[idx])
    for v_idx in range(n_variants):
        axs[0].plot(hybrids_g1[v_idx], target_depths, color=variant_colors[v_idx], alpha=0.9, linewidth=1.6, label=f'Вариант {v_idx+1}')
    axs[0].invert_yaxis()
    axs[0].set_title("Группа 1", fontsize=12, weight='bold', pad=10)
    axs[0].set_xlabel(f"Значения {method_name}, {unit_name}", fontsize=11)
    axs[0].set_ylabel("Глубина, м", fontsize=11)
    axs[0].grid(True, linestyle=':', alpha=0.4)
    if x_limits:
        axs[0].set_xlim(x_limits)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, facecolor='white', edgecolor='none', fontsize=9)
    
    # Правый планшет (Группа 2)
    for c_idx, idx in enumerate(g2_idx):
        color_idx = (c_idx + len(g1_idx)) % len(all_colors)
        axs[1].plot(data_raw[idx], X_depths[idx], color=all_colors[color_idx], alpha=0.25, linewidth=1.0, label=well_names[idx])
    for v_idx in range(n_variants):
        axs[1].plot(hybrids_g2[v_idx], target_depths, color=variant_colors[v_idx], alpha=0.9, linewidth=1.6, label=f'Вариант {v_idx+1}')
    axs[1].invert_yaxis()
    axs[1].set_title("Группа 2", fontsize=12, weight='bold', pad=10)
    axs[1].set_xlabel(f"Значения {method_name}, {unit_name}", fontsize=11)
    axs[1].grid(True, linestyle=':', alpha=0.4)
    if x_limits:
        axs[1].set_xlim(x_limits)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, facecolor='white', edgecolor='none', fontsize=9)
        
    plt.suptitle(f"CVAE-моделирование ({method_name})\nПласт ТП3 Фация 2", fontsize=13, y=0.96, weight='bold')
    
    target_path = f"{output_dir}/cvae_plate_{file_label}.png"
    plt.savefig(target_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Планшет [{method_name}] успешно сохранен в: {target_path}")

draw_cvae_plate(X_gk, hybrids_gk_g1, hybrids_gk_g2, "ГК", "мкР/ч", "GK", x_limits=(2.5, 10.5))
draw_cvae_plate(X_ps, hybrids_ps_g1, hybrids_ps_g2, "ПС", "мВ", "PS", x_limits=(-15, 130))
print("Все расчеты и сохранение планшетов завершены!")