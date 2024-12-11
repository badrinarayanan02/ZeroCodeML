import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="ZeroCodeML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a colorful background style
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(90deg, #ff7eb3, #ff758c);
        color: #ffffff;
    }
    .css-18e3th9 {
        background-color: rgba(0, 0, 0, 0.5);
    }
    .stButton>button {
        background-color: #ff758c;
        color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header with dynamic styling
st.title("🚀 ZeroCodeML: No Code Machine Learning Tool")
st.caption(
    "A web-based application for creating machine learning models without coding, designed for non-expert users!"
)

# About Page Content
st.subheader("Welcome to ZeroCodeML!")
st.markdown(
    """
    ### About ZeroCodeML
    ZeroCodeML simplifies the process of creating machine learning models, allowing non-experts to build, analyze, and deploy models with ease.

    ### Features
    - **EDA**: Explore and visualize your dataset.
    - **Modelling**: Train and evaluate ML models for classification or regression tasks.
    - **Prediction**: Predict new data using a trained model.
    
    ### Why Choose ZeroCodeML?
    - Accessible from anywhere via the internet.
    - No software installation required.
    - Interactive and real-time learning experience.
    """
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ by ZeroCodeML Team.")

