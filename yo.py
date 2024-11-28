import streamlit as st
try:
    import faiss
    st.success("FAISS is installed and working!")
except ModuleNotFoundError as e:
    st.error(f"FAISS is not installed: {e}")
