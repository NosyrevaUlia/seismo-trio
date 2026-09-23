import streamlit as st
import sys
import os

# Добавляем папку geosynth в пути поиска, чтобы можно было импортировать app_geo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'geosynth'))

# Импортирт функций из наших приложений
from app_seis import run_seismic_app
from app_geo_cvae import run_geo_app

st.set_page_config(
    page_title="Модули генерации",
    layout="wide"
)

# Инициализация роутера
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
        if st.button("Запустить Сейсмику", use_container_width=True):
            st.session_state.current_page = 'seis'
            st.rerun()
            
    with col2:
        st.markdown("### GeoSynth")
        st.markdown("Генерация каротажных данных.")
        if st.button("Запустить Геофизику", use_container_width=True):
            st.session_state.current_page = 'geo'
            st.rerun()

elif st.session_state.current_page == 'seis':
    run_seismic_app()
    
elif st.session_state.current_page == 'geo':
    run_geo_app()