import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix,roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# Load dataset with caching for performance
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/abod\Downloads/Customer_churn_raw.csv")
    return df

df = load_data()

# Sidebar options for navigation
st.sidebar.title("Customer Churn Analysis")
option = st.sidebar.selectbox("Select Analysis Stage", [
    "Data Overview", "Data Preprocessing", "Exploratory Data Analysis", "Feature Engineering", "Model Training", "Hyperparameter Tuning", "Model Evaluation", "Predictions"
])

if option == "Data Overview":
    st.title("Dataset Overview")
    st.write(df.head())  # Display first few rows of the dataset
    st.write("### Data Summary")
    st.write(df.describe())  # Show dataset statistics
    st.write("### Missing Values")
    st.write(df.isnull().sum())  # Display count of missing values per column
    
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)  # Visualize class distribution
    st.pyplot(fig)

elif option == "Data Preprocessing":
    st.title("Data Preprocessing")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Handle missing values separately
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Ensure all numeric columns are properly converted
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill missing values in numeric columns with median
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])  # Fill categorical columns with mode
    
    # Encode categorical variables using Label Encoding
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert categorical to string before encoding
    
    # Scale numerical features for uniformity
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save preprocessed data
    df.to_csv('c:/Users/abod\Downloads/code/preprocessed.csv', index=False)
    
    st.write("Preprocessing Complete!")
    st.write(df.head())

elif option == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    st.write("### Correlation Matrix")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Select only numerical columns
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')  # Display feature correlations
    st.pyplot(fig)

elif option == "Feature Engineering":
    st.title("Feature Engineering")
    
    # Debugging: Show available columns before feature engineering
    st.write("### Available Columns:")
    st.write(df.columns.tolist())
    
    # Ensure correct column names are used for feature engineering
    if 'Charge  Amount' in df.columns and 'Subscription  Length' in df.columns:
        # Convert columns to numeric to avoid type errors
        df['Charge  Amount'] = pd.to_numeric(df['Charge  Amount'], errors='coerce')
        df['Subscription  Length'] = pd.to_numeric(df['Subscription  Length'], errors='coerce')
        
        # Handle missing values
        df['Charge  Amount'].fillna(df['Charge  Amount'].median(), inplace=True)
        df['Subscription  Length'].fillna(df['Subscription  Length'].median(), inplace=True)
        
        # Create new feature
        df['TotalCharges'] = df['Charge  Amount'] * df['Subscription  Length']
        
        st.write("Feature Engineering Complete! New features added.")
        st.write(df.head())
    else:
        st.write("Error: 'Charge  Amount' or 'Subscription  Length' column is missing!")

# Define features (X) and target variable (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Encode the target variable as discrete classes
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in training data
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Save the split datasets
pd.DataFrame(X_train, columns=X.columns).to_csv('c:/Users/abod\Downloads/code/train.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv('c:/Users/abod\Downloads/code/test.csv', index=False)
pd.DataFrame(y_train, columns=['Churn']).to_csv('c:/Users/abod\Downloads/code/y_train.csv', index=False)
pd.DataFrame(y_test, columns=['Churn']).to_csv('c:/Users/abod\Downloads/code/y_test.csv', index=False)

# Save the selected features
selected_features = X.columns.tolist()
pd.Series(selected_features).to_csv('c:/Users/abod\Downloads/code/selected_features.csv', index=False)

if option == "Model Training":
    st.title("Train Machine Learning Model")
    
    # Train a Random Forest model on balanced data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Save the trained model
        joblib.dump(model, 'model.pkl')
        
        st.write("### Model Training Complete")

    except Exception as e:
        st.error(f"An error occurred during model training: {e}")

elif option == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    
    try:
        grid_search.fit(X_train, y_train)  
        best_params = grid_search.best_params_  
        
        st.write("Best Parameters:")
        st.write(best_params)
        
        model = grid_search.best_estimator_
        model.fit(X_train, y_train)
        
        joblib.dump(model, 'model.pkl')

    except Exception as e:
        st.error(f"An error occurred during hyperparameter tuning: {e}")


# Assuming y_test and y_train are defined somewhere in your code

elif option == "Model Evaluation":
    st.title("Model Evaluation")
    
    try:
        model = joblib.load('model.pkl')
        
        if model is None:
            st.write("Model is not trained yet. Please train the model first.")
        else:
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            st.write(f"Accuracy: {accuracy:.2f}")  
            st.write(f"AUC: {auc:.2f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))  
            
            fig1, ax1 = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')  
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')
            st.pyplot(fig1)

            ### Histogram for Churn Distribution Before and After Balancing ###
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            
            sns.countplot(x=pd.Series(y_test).astype(str), ax=ax2[0])
            ax2[0].set_title('Churn Distribution Before Balancing')
            
            sns.countplot(x=pd.Series(y_train).astype(str), ax=ax2[1])
            ax2[1].set_title('Churn Distribution After Balancing')

            for a in ax2:
                a.set_xlabel('Churn')
                a.set_ylabel('Count')

            st.pyplot(fig2)

            ### Clustered Bar Chart for Accuracy and AUC Before and After Training ###
            metrics_before = {'Metric': ['Accuracy', 'AUC'], 'Value': [0.5, 0.5], 'Type': 'Before Training'}
            metrics_after = {'Metric': ['Accuracy', 'AUC'], 'Value': [accuracy, auc], 'Type': 'After Training'}

            metrics_df = pd.concat([pd.DataFrame(metrics_before), pd.DataFrame(metrics_after)], ignore_index=True)

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Metric', y='Value', hue='Type', data=metrics_df, ax=ax3)
            ax3.set_title('Accuracy and AUC Before and After Training')
            st.pyplot(fig3)

    except FileNotFoundError:
        model = None

if option == "Predictions":
    st.title("Make Predictions")

    try:
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        model = None

    if model is None:
        st.write("Model is not trained yet. Please train the model first.")
    else:
        st.write("### Enter Feature Values")

        input_data = {}
        predefined_values = {
            'feature1': [0],  
            'feature2': [0],  
            'feature3': [10],  
        }

        for feature in selected_features:
            if feature in predefined_values:
                input_data[feature] = st.selectbox(f"Select value for {feature}", predefined_values[feature])
            else:
                input_data[feature] = st.number_input(f"Enter value for {feature}", value=None)

        input_df = pd.DataFrame([input_data])

        input_df[selected_features] = input_df[selected_features].apply(pd.to_numeric)

        # Apply imputer transformation directly to input_df
        input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

        predictions = model.predict(input_df)

        st.write("### Churn Prediction")
        st.write(predictions[0])