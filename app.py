import streamlit as st
import sys
import os

from app_seis import run_seismic_app
from app_geo_cvae import run_geo_app

# --- ВСТАВИТЬ ПОСЛЕ ИМПОРТОВ В app.py ---
st.markdown("""
<style>
/* Импорт шрифтов */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@600;700;800&display=swap');

    /* 1. Уменьшаем базовый текст и интерфейс */
    html, body, [class*="css"], p, label, .stMarkdown, input, select, textarea, div[data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important; /* Базовый размер уменьшен до 14px */
    }

    /* 2. Настраиваем заголовки (делаем их аккуратнее) */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif !important;
        letter-spacing: -0.5px;
    }
    h1 { font-size: 2rem !important; }   /* Главный заголовок */
    h2 { font-size: 1.5rem !important; } /* Заголовки карточек */
    h3 { font-size: 1.2rem !important; }
    h4 { font-size: 1.1rem !important; }

    /* 3. Уменьшаем шрифт в кнопках */
    .stButton > button, div.stDownloadButton > button {
        font-size: 14px !important;
    }
    
    /* 4. Уменьшаем размер текста в таблицах и метриках */
    .stDataFrame, .stTable {
        font-size: 13px !important;
    }
    
    /* 1. Небесный градиентный фон (Голубой -> Лавандовый) */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #E0E7FF 0%, #F3E8FF 45%, #DBEAFE 100%);
        background-attachment: fixed;
    }

    /* 2. Матовое стекло для боковой панели с легким фиолетовым отливом */
    [data-testid="stSidebar"] {
        background: rgba(243, 232, 255, 0.5) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }

    /* 3. Градиентный текст для главного заголовка H1 (Голубой -> Фиолетовый) */
    h1 {
        background: -webkit-linear-gradient(45deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* 4. Кнопки (Нежный градиент) */
    .stButton > button {
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 16px -6px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s ease !important;
        padding: 0.5rem 1.5rem !important;
    }

    /* Эффект при наведении (кнопка становится чуть ярче и приподнимается) */
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 20px -6px rgba(139, 92, 246, 0.6) !important;
        filter: brightness(1.05) !important;
    }

    /* 5. Стеклянные карточки (Воздушные, с фиолетовой тенью) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 1) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px -20px rgba(139, 92, 246, 0.2) !important;
        padding: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 45px -20px rgba(59, 130, 246, 0.3) !important;
        border: 1px solid rgba(167, 139, 250, 0.4) !important;
    }

    /* 6. Блоки настроек в сайдбаре */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
        border-radius: 14px !important;
    }
    
        /* 8. Специальные стили для кнопок экспорта (st.download_button) */
    div.stDownloadButton > button {
        /* Градиент от насыщенного фиолетового к яркому голубому */
        background: linear-gradient(135deg, #A78BFA 0%, #60A5FA 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        position: relative;
    }

    /* Эффект "оживания" при наведении */
    div.stDownloadButton > button:hover {
        /* Кнопка приподнимается и немного увеличивается */
        transform: translateY(-3px) scale(1.02) !important; 
        /* Усиливаем тень (свечение) */
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
        /* Делаем градиент чуть более контрастным */
        background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
        color: #ffffff !important;
    }

    /* Эффект нажатия (кнопка словно вдавливается) */
    div.stDownloadButton > button:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 4px 10px rgba(139, 92, 246, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)
# -------------------------------------
st.set_page_config(
    page_title="SynthGeoGen",
    layout="wide"
)

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Переключение страниц
if st.session_state.current_page == 'home':
    st.markdown("<h3 style='text-align: center; color: gray;'>Выберите приложение для запуска</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Генератор сейсмических трасс")
        st.markdown("Синтез, просмотр и анализ сейсмических данных (SEG-Y).")
        if st.button("Запустить", use_container_width=True, key="btn_start_seis"):
            st.session_state.current_page = 'seis'
            st.rerun()
            
    with col2:
        st.markdown("### GeoSynth")
        st.markdown("Генерация каротажных данных.")
        if st.button("Запустить", use_container_width=True, key="btn_start_geo"):
            st.session_state.current_page = 'geo'
            st.rerun()

elif st.session_state.current_page == 'seis':
    run_seismic_app()
    
elif st.session_state.current_page == 'geo':
    run_geo_app()