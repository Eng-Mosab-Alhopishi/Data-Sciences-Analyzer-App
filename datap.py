import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from sklearn.impute import SimpleImputer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_session_state():
    session_vars = [
        'data', 'processed_data', 'model', 'label_encoder',
        'selected_features', 'target_column', 'test_size',
        'show_splash', 'num_classes', 'class_distribution'
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    if 'show_splash' not in st.session_state:
        st.session_state.show_splash = True

def inject_custom_css():
    st.markdown(f"""
    <style>
    :root {{
        --primary: {st.get_option('theme.primaryColor')};
        --background: {st.get_option('theme.backgroundColor')};
        --secondary-bg: {st.get_option('theme.secondaryBackgroundColor')};
        --text: {st.get_option('theme.textColor')};
    }}

    .main {{
        background-color: var(--background);
        color: var(--text);
        padding-bottom: 100px;
        position: relative;
        z-index: 1;
    }}

    .splash-screen {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: var(--background);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        opacity: 1;
        transition: opacity 0.5s ease-out;
    }}

    .splash-hidden {{
        opacity: 0;
        pointer-events: none;
    }}

    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: var(--secondary-bg);
        padding: 15px;
        text-align: center;
        border-top: 2px solid var(--primary);
        font-family: 'Arial', sans-serif;
        font-size: 0.9em;
        z-index: 0;
    }}

    .card {{
        background: var(--secondary-bg);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .loading-spinner {{
        width: 50px;
        height: 50px;
        border: 5px solid var(--primary);
        border-radius: 50%;
        border-top-color: transparent;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Data Analyzer Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    inject_custom_css()

    if st.session_state.show_splash:
        show_splash_screen()
        time.sleep(2)
        st.session_state.show_splash = False
        st.rerun()

    st.title("🤖 AI-Powered Data Analysis & Modeling Pro")
    
    st.markdown("""
    <div class="footer">
        🚀 Developed by <span style="color: var(--primary); font-weight: bold;">ENG - MOSAB AL-hopishi</span> | 
        📧 Contact: example@email.com | 
        🔧 Version 5.1
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "📁 Data Upload": data_upload,
        "⚙️ Processing": data_processing,
        "📊 Visualization": data_visualization,
        "🧠 Model Training": model_training,
        "🔮 Prediction": prediction
    }
    
    selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    pages[selected_page]()

def show_splash_screen():
    st.markdown("""
    <div class="splash-screen">
        <div style="text-align: center;">
            <h1 style="color: var(--primary); margin-bottom: 30px; font-size: 2.5em;">🧠 AI Data Analyzer Pro</h1>
            <div class="loading-spinner"></div>
            <p style="color: var(--text); margin-top: 20px; font-size: 1.2em;">Initializing Advanced Components...</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def data_upload():
    st.header("📁 Data Upload")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=["csv", "xlsx"],
            help="Max file size: 200MB"
        )

    if uploaded_file:
        try:
            with st.spinner('Analyzing file structure...'):
                start_time = time.time()
                
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.session_state.class_distribution = df[df.columns[-1]].value_counts().to_dict()
                
                st.success(f"✅ Successfully loaded {len(df)} records")
                
                cols = st.columns(4)
                with cols[0]:
                    with st.container(border=True):
                        st.metric("Total Features", df.shape[1])
                with cols[1]:
                    with st.container(border=True):
                        st.metric("Numeric Features", len(df.select_dtypes(include=np.number).columns))
                with cols[2]:
                    with st.container(border=True):
                        st.metric("Categorical Features", len(df.select_dtypes(exclude=np.number).columns))
                with cols[3]:
                    with st.container(border=True):
                        st.metric("Missing Values", df.isnull().sum().sum())

                with st.expander("🔍 Preview First 10 Rows", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.data = None

def data_processing():
    st.header("⚙️ Data Processing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data

    with st.form("processing_form"):
        st.subheader("Processing Configuration")
        
        target_col = st.selectbox(
            "🎯 Select Target Column", 
            df.columns,
            help="The variable you want to predict"
        )
        st.session_state.target_column = target_col

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🧼 Cleaning Options")
                handle_missing = st.selectbox(
                    "Missing Values Handling",
                    ["Drop rows", "Drop columns", "Impute (future)"],
                    index=0,
                    disabled=True
                )
                
                encode_cat = st.checkbox(
                    "Encode Categorical Features", 
                    value=True,
                    help="Auto-encode non-numeric columns"
                )

        with col2:
            with st.container(border=True):
                st.markdown("### ⚖️ Advanced Balancing")
                
                if st.session_state.class_distribution:
                    st.write("**Class Distribution:**")
                    for cls, count in st.session_state.class_distribution.items():
                        st.write(f"- Class {cls}: {count} samples")
                
                if st.session_state.num_classes and st.session_state.num_classes > 2:
                    balance_options = [
                        "None",
                        "RandomOverSampler",
                        "SMOTE",
                        "SMOTE + OverSampler"
                    ]
                else:
                    balance_options = [
                        "None",
                        "RandomOverSampler",
                        "RandomUnderSampler",
                        "SMOTE",
                        "SMOTE + UnderSampling"
                    ]
                    
                balance_method = st.selectbox(
                    "Balance Method",
                    balance_options,
                    help="Auto-adjusted based on class distribution"
                )
                
                st.markdown("### 🔍 Feature Selection")
                k_features = st.slider(
                    "Select Top Features",
                    min_value=2,
                    max_value=min(20, df.shape[1]-1),
                    value=10,
                    help="ANOVA F-value based selection"
                )

        with st.expander("⚙️ Advanced Settings"):
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20
            )
            st.session_state.test_size = test_size / 100

        if st.form_submit_button("🚀 Start Processing", use_container_width=True):
            try:
                with st.spinner('Processing data...'):
                    start_time = time.time()
                    process_data(df, target_col, k_features, balance_method, encode_cat)
                    process_time = time.time() - start_time
                    
                    st.toast(f'🎉 Processing completed in {process_time:.2f}s!', icon='✅')
                    display_processing_results()

            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")

    if st.session_state.processed_data is not None:
        with st.container(border=True):
            st.markdown("### 📤 Export Processed Data")
            csv = st.session_state.processed_data['dataframe'].to_csv(index=False).encode()
            st.download_button(
                "💾 Download Processed Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )

def process_data(df, target_col, k_features, balance_method, encode_cat):
    processed_df = df.copy()
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='median')
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])
    
    # Encode target
    le = LabelEncoder()
    processed_df[target_col] = le.fit_transform(processed_df[target_col])
    st.session_state.label_encoder = le
    st.session_state.num_classes = len(np.unique(processed_df[target_col]))
    
    # Encode categorical features
    if encode_cat:
        cat_cols = processed_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != target_col:
                processed_df[col] = le.fit_transform(processed_df[col])
    
    # Split features and target
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    
    # Standardize numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Balance classes
    if balance_method != "None":
        sampling_strategy = get_sampling_strategy(y, balance_method)
        
        if balance_method == "RandomOverSampler":
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X, y = ros.fit_resample(X, y)
            
        elif balance_method == "RandomUnderSampler":
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X, y = rus.fit_resample(X, y)
            
        elif balance_method == "SMOTE":
            X, y = self_healing_smote(X, y, sampling_strategy)
            
        elif balance_method == "SMOTE + OverSampler":
            pipeline = Pipeline([
                ('smote', SMOTE(
                    sampling_strategy=0.75,
                    k_neighbors=dynamic_neighbors(y),
                    random_state=42
                )),
                ('over', RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                ))
            ])
            X, y = pipeline.fit_resample(X, y)
            
        elif balance_method == "SMOTE + UnderSampling":
            pipeline = Pipeline([
                ('smote', SMOTE(
                    sampling_strategy=0.75,
                    k_neighbors=dynamic_neighbors(y),
                    random_state=42
                )),
                ('under', RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                ))
            ])
            X, y = pipeline.fit_resample(X, y)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, 
        test_size=st.session_state.test_size,
        random_state=42
    )
    
    # Store processed data
    st.session_state.processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': selected_features,
        'dataframe': pd.DataFrame(X_selected, columns=selected_features)
    }

def get_sampling_strategy(y, balance_method):
    unique, counts = np.unique(y, return_counts=True)
    majority_count = max(counts)
    strategy = {}

    for cls, count in zip(unique, counts):
        if balance_method in ["SMOTE", "SMOTE + OverSampler", "SMOTE + UnderSampling"]:
            min_samples_required = 6
            if count < min_samples_required:
                strategy[cls] = count
                st.warning(f"⚠️ Class {cls} has only {count} samples. Cannot apply {balance_method}.")
            else:
                strategy[cls] = majority_count if balance_method != "RandomUnderSampler" else min(counts)
        elif balance_method == "RandomOverSampler":
            strategy[cls] = majority_count
        elif balance_method == "RandomUnderSampler":
            strategy[cls] = min(counts)
    
    return strategy

def dynamic_neighbors(y):
    counts = np.bincount(y)
    min_samples = min(counts[counts > 0])
    return min(5, min_samples - 1) if min_samples > 1 else 1

def self_healing_smote(X, y, sampling_strategy):
    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=dynamic_neighbors(y),
            random_state=42
        )
        return smote.fit_resample(X, y)
    except ValueError as e:
        if "Expected n_neighbors <= n_samples" in str(e):
            st.warning("⚠️ Adjusting SMOTE parameters due to small sample size")
            new_strategy = {k: v for k, v in sampling_strategy.items() if v >= 6}
            return X, y
        raise e

def display_processing_results():
    data = st.session_state.processed_data
    
    st.subheader("📈 Processing Results")
    cols = st.columns(4)
    with cols[0]:
        with st.container(border=True):
            st.metric("Selected Features", len(data['features']))
    with cols[1]:
        with st.container(border=True):
            st.metric("Training Samples", data['X_train'].shape[0])
    with cols[2]:
        with st.container(border=True):
            st.metric("Test Samples", data['X_test'].shape[0])
    with cols[3]:
        with st.container(border=True):
            st.metric("Class Balance", 
                    f"{np.unique(data['y_train'], return_counts=True)[1][0]/len(data['y_train']):.1%}")

    with st.expander("📋 Feature List", expanded=False):
        st.write(data['features'].tolist())

def data_visualization():
    st.header("📊 Data Visualization")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first!")
        return
    
    data = st.session_state.processed_data
    df = data['dataframe']
    features = data['features']

    viz_type = st.selectbox(
        "Choose Visualization Type",
        ["📈 Distribution Plot", "🌡️ Correlation Heatmap", "🔗 Pair Plot", "🕹️ 3D Scatter"]
    )

    if viz_type == "📈 Distribution Plot":
        selected_feature = st.selectbox("Select Feature", features)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        st.pyplot(fig)

    elif viz_type == "🌡️ Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif viz_type == "