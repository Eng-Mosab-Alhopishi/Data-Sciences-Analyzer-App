import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تهيئة حالة الجلسة
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'balance_method' not in st.session_state:
    st.session_state.balance_method = "SMOTE"
if 'class_distribution' not in st.session_state:
    st.session_state.class_distribution = {}

def main():
    st.title("🏢 Employee Attrition Prediction System")
    
    menu_options = {
        "📁 Upload Data": upload_data,
        "🔍 Data Analysis": data_analysis,
        "⚙️ Data Processing": data_processing,
        "🤖 Train Model": train_model,
        "🔮 Predict": predict
    }
    
    page = st.sidebar.selectbox("Navigation", list(menu_options.keys()))
    menu_options[page]()

def upload_data():
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload HR Dataset (CSV)", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # التحقق من وجود عمود الهدف
            if 'Attrition' not in df.columns:
                st.error("Column 'Attrition' not found in dataset!")
                return
                
            # تحويل الهدف إلى رقمي
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0}).fillna(df['Attrition'])
            df['Attrition'] = df['Attrition'].astype(int)
            
            st.session_state.data = df
            st.session_state.class_distribution = df['Attrition'].value_counts().to_dict()
            
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # عرض معلومات أساسية
            cols = st.columns(4)
            attrition_rate = df['Attrition'].mean()
            cols[0].metric("Total Employees", df.shape[0])
            cols[1].metric("Attrition Rate", f"{attrition_rate:.2%}")
            cols[2].metric("Features", df.shape[1])
            cols[3].metric("Missing Values", df.isnull().sum().sum())
            
            st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def data_analysis():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    st.header("Data Analysis")
    
    try:
        # عرض توزيع الفئات
        with st.expander("📊 Class Distribution"):
            fig, ax = plt.subplots()
            df['Attrition'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Attrition Distribution')
            ax.set_xlabel('Attrition Status')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        # تحليل توزيع الأعمار
        with st.expander("📈 Age Distribution"):
            if 'Age' in df.columns:
                fig, ax = plt.subplots()
                sns.histplot(df[df['Attrition'] == 1]['Age'], bins=20, kde=True, color='red', label='Attrition', ax=ax)
                sns.histplot(df[df['Attrition'] == 0]['Age'], bins=20, kde=True, color='green', label='No Attrition', ax=ax)
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Age column not found in data")
        
        # تحليل معدل التنازل حسب القسم
        with st.expander("📉 Attrition by Department"):
            if 'Department' in df.columns:
                department_stats = df.groupby('Department')['Attrition'].mean().reset_index()
                fig, ax = plt.subplots()
                sns.barplot(x='Department', y='Attrition', data=department_stats, ax=ax)
                ax.set_title('Attrition Rate by Department')
                ax.set_ylabel('Attrition Rate')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Department column not found in data")
        
        # مصفوفة الارتباط
        with st.expander("🔗 Correlation Matrix"):
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(12,8))
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns for correlation analysis")
                
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

def data_processing():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("Data Processing")
    
    with st.form("processing_form"):
        st.subheader("Balancing Configuration")
        
        # اختيار طريقة الموازنة
        balance_method = st.selectbox(
            "Select Balancing Method",
            ["SMOTE", "RandomOverSampler", "None"],
            index=0
        )
        
        st.session_state.balance_method = balance_method
        
        if st.form_submit_button("Apply Processing"):
            try:
                df = st.session_state.data.copy()
                X = df.drop('Attrition', axis=1)
                y = df['Attrition']
                
                # تطبيق طريقة الموازنة المختارة
                if balance_method != "None":
                    if balance_method == "SMOTE":
                        smote = SMOTE(random_state=42)
                        X_res, y_res = smote.fit_resample(X, y)
                    elif balance_method == "RandomOverSampler":
                        from imblearn.over_sampling import RandomOverSampler
                        ros = RandomOverSampler(random_state=42)
                        X_res, y_res = ros.fit_resample(X, y)
                    
                    st.session_state.class_distribution = pd.Series(y_res).value_counts().to_dict()
                    st.success(f"Balancing applied using {balance_method}. New distribution: {st.session_state.class_distribution}")
                else:
                    st.session_state.class_distribution = pd.Series(y).value_counts().to_dict()
                    st.success("Using original imbalanced data")
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

def preprocess_data():
    try:
        df = st.session_state.data.copy()
        
        # معالجة البيانات
        df = df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, errors='ignore')
        
        # ترميز الفئات
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col != 'Attrition':
                df[col] = df[col].astype('category').cat.codes
        
        # التعامل مع القيم المفقودة
        df = df.fillna(df.median(numeric_only=True))
        
        # موازنة البيانات
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        
        if st.session_state.balance_method != "None":
            if st.session_state.balance_method == "SMOTE":
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
            elif st.session_state.balance_method == "RandomOverSampler":
                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(random_state=42)
                X_res, y_res = ros.fit_resample(X, y)
        else:
            X_res, y_res = X, y
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, 
            test_size=0.2, 
            random_state=42,
            stratify=y_res
        )
        
        # تطبيع البيانات
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None, None, None, None, None

def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    st.header("Model Training")
    
    with st.spinner('Preprocessing data...'):
        X_train, X_test, y_train, y_test, scaler = preprocess_data()
        
    if X_train is None:
        return
    
    # إعداد النموذج
    with st.form("training_form"):
        st.subheader("Model Configuration")
        
        model_type = st.selectbox("Select Model Type", ["Random Forest", "Neural Network"])
        
        if model_type == "Random Forest":
            cols = st.columns(2)
            n_estimators = cols[0].slider("Number of Trees", 50, 500, 200)
            max_depth = cols[1].slider("Max Depth", 2, 30, 10)
            class_weight = cols[0].selectbox("Class Weight", ["balanced", "None"])
            bootstrap = cols[1].checkbox("Bootstrap", value=True)
        else:
            epochs = st.slider("Epochs", 10, 100, 50)
            batch_size = st.slider("Batch Size", 16, 128, 32)
        
        if st.form_submit_button("Train Model"):
            try:
                if model_type == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight='balanced' if class_weight == "balanced" else None,
                        bootstrap=bootstrap,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                else:
                    model = build_nn_model(X_train.shape[1])
                    history = model.fit(
                        X_train, y_train,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                    st.session_state.history = history
                
                st.session_state.model = model
                st.session_state.scaler = scaler
                evaluate_model(model, X_test, y_test)
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")

def evaluate_model(model, X_test, y_test):
    try:
        st.subheader("Model Evaluation")
        
        if isinstance(model, Sequential):  # Neural Network
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype(int)
            y_proba = y_pred_probs.flatten()
        else:  # Random Forest
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # حساب المقاييس
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba)
        }
        
        # عرض النتائج
        cols = st.columns(5)
        for i, (name, value) in enumerate(metrics.items()):
            cols[i].metric(name, f"{value:.2%}" if isinstance(value, float) else value)
        
        # عرض مصفوفة الارتباك
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        # عرض تقرير التصنيف
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2%}"))
        
        # عرض العوامل المؤثرة (لـ Random Forest فقط)
        if not isinstance(model, Sequential):
            st.subheader("Top Important Features")
            feature_importance = pd.Series(model.feature_importances_, 
                                         index=st.session_state.data.drop('Attrition', axis=1).columns)
            top_features = feature_importance.sort_values(ascending=False)[:10]
            st.bar_chart(top_features)
        
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")

def predict():
    if st.session_state.model is None:
        st.warning("Please train model first!")
        return
    
    st.header("Predict Attrition")
    df = st.session_state.data
    
    try:
        # نموذج إدخال البيانات
        inputs = {}
        features = df.drop('Attrition', axis=1).columns
        
        cols = st.columns(3)
        for i, col in enumerate(features):
            with cols[i%3]:
                if df[col].dtype == 'object':
                    inputs[col] = st.selectbox(col, df[col].unique())
                else:
                    inputs[col] = st.number_input(col, value=df[col].median())
        
        if st.button("Predict"):
            # تحويل المدخلات
            input_df = pd.DataFrame([inputs])
            cat_cols = input_df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                input_df[col] = input_df[col].astype('category').cat.codes
            
            # تطبيق التطبيع
            scaled_input = st.session_state.scaler.transform(input_df)
            
            # التنبؤ
            model = st.session_state.model
            if isinstance(model, Sequential):  # Neural Network
                prediction_proba = model.predict(scaled_input)[0][0]
                prediction = 1 if prediction_proba > 0.5 else 0
            else:  # Random Forest
                prediction = model.predict(scaled_input)[0]
                prediction_proba = model.predict_proba(scaled_input)[0][1]
            
            # عرض النتائج
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"🔥 High Risk of Attrition ({prediction_proba:.2%} probability)")
            else:
                st.success(f"✅ Low Risk of Attrition ({1-prediction_proba:.2%} probability)")
                
            # عرض تفسير النموذج (لـ Random Forest فقط)
            if not isinstance(model, Sequential):
                st.subheader("Key Influencing Factors")
                feature_importance = pd.Series(model.feature_importances_, 
                                            index=df.drop('Attrition', axis=1).columns)
                top_features = feature_importance.sort_values(ascending=False)[:3]
                
                for feature, importance in top_features.items():
                    value = inputs[feature]
                    st.write(f"• **{feature}**: {value} (Impact: {importance:.2f})")
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()

