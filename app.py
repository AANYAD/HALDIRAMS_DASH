"""
================================================================================
Haldiram's International Export Analytics Pipeline - Entry Point
================================================================================
Author: Aanya
Project: Haldiram's Export Sales & ML Analytics
================================================================================
"""

import streamlit as st

st.set_page_config(
    page_title="Haldiram's International Export Analytics",
    page_icon="assets/haldirams_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.switch_page("pages/0_📊_Overview_&_EDA.py")
