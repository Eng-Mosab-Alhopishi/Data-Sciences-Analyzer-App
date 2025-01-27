import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing and Visualization App")
        self.root.geometry("1000x700")

        self.data = None
        self.processed_data = None
        self.target_column = None
        self.model = None
        self.train_data, self.test_data, self.train_labels, self.test_labels = None, None, None, None
        self.selected_features = None
        self.features_before = None
        self.features_after = None
        self.data_selected = None
        self.label_encoder = None  # To store LabelEncoder for target column

        self.setup_ui()

    def setup_ui(self):
        # Styling
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 10), padding=5)
        self.style.configure("TLabel", font=("Arial", 10), background="#f0f0f0")

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        # Tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.processing_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.processing_tab, text="Processing")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.prediction_tab, text="Prediction")

        self.setup_data_tab()
        self.setup_processing_tab()
        self.setup_visualization_tab()
        self.setup_model_tab()
        self.setup_prediction_tab()

    def setup_data_tab(self):
        tk.Button(self.data_tab, text="Load Data", command=self.load_data).pack(pady=10)
        self.data_log = tk.Text(self.data_tab, height=10, state='disabled')
        self.data_log.pack(fill='both', expand=True)

    def setup_processing_tab(self):
        tk.Label(self.processing_tab, text="Select Target Column:").pack(pady=10)
        self.target_column_combo = ttk.Combobox(self.processing_tab, state='readonly')
        self.target_column_combo.pack()

        tk.Button(self.processing_tab, text="Start Processing", command=self.start_processing).pack(pady=10)
        self.process_log = tk.Text(self.processing_tab, height=10, state='disabled')
        self.process_log.pack(fill='both', expand=True)

        tk.Button(self.processing_tab, text="Save Clean Data", command=self.save_clean_data).pack(pady=10)
        tk.Button(self.processing_tab, text="Check Processing Steps", command=self.check_processing_steps).pack(pady=10)

    def setup_visualization_tab(self):
        tk.Label(self.visualization_tab, text="Select Features to Plot (Max 3):").pack(pady=10)
        self.feature_list = tk.Listbox(self.visualization_tab, selectmode='multiple')
        self.feature_list.pack()

        tk.Button(self.visualization_tab, text="Plot Data", command=self.plot_data).pack(pady=10)

    def setup_model_tab(self):
        tk.Label(self.model_tab, text="Select Model:").pack(pady=10)
        self.model_combo = ttk.Combobox(self.model_tab, state='readonly', values=["Random Forest", "Deep Learning"])
        self.model_combo.pack()

        tk.Button(self.model_tab, text="Train Model", command=self.train_model).pack(pady=10)
        tk.Button(self.model_tab, text="Test Model", command=self.test_model).pack(pady=10)
        self.model_log = tk.Text(self.model_tab, height=10, state='disabled')
        self.model_log.pack(fill='both', expand=True)

    def setup_prediction_tab(self):
        tk.Label(self.prediction_tab, text="Enter Feature Values for Prediction:").pack(pady=10)
        self.prediction_entries = []
        self.prediction_labels = []
        for i in range(10):  # Assuming max 10 features
            frame = ttk.Frame(self.prediction_tab)
            frame.pack(fill='x', padx=10, pady=5)
            label = ttk.Label(frame, text=f"Feature {i+1}:")
            label.pack(side='left')
            entry = ttk.Entry(frame)
            entry.pack(side='right', fill='x', expand=True)
            self.prediction_entries.append(entry)
            self.prediction_labels.append(label)

        tk.Button(self.prediction_tab, text="Predict", command=self.predict).pack(pady=10)
        tk.Button(self.prediction_tab, text="New Prediction", command=self.new_prediction).pack(pady=10)  # New button
        self.prediction_result = ttk.Label(self.prediction_tab, text="Prediction result will appear here.")
        self.prediction_result.pack()

    def log(self, message, log_widget):
        log_widget.configure(state='normal')
        log_widget.insert('end', message + '\n')
        log_widget.configure(state='disabled')
        log_widget.see('end')

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)

            self.log("Data loaded successfully!", self.data_log)
            self.log(f"Columns: {', '.join(self.data.columns)}", self.data_log)
            self.target_column_combo['values'] = self.data.columns.tolist()
            self.feature_list.delete(0, 'end')
            for column in self.data.columns:
                self.feature_list.insert('end', column)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def start_processing(self):
        if self.data is None or self.data.empty:
            self.log("No data to process. Please upload a file.", self.process_log)
            return

        self.target_column = self.target_column_combo.get()
        if not self.target_column:
            self.log("No target column selected.", self.process_log)
            return

        try:
            # Drop rows with missing values
            self.data.dropna(inplace=True)
            self.log("Dropped rows with missing values.", self.process_log)

            # Encode target column
            self.label_encoder = LabelEncoder()  # Save LabelEncoder for later use
            self.data[self.target_column] = self.label_encoder.fit_transform(self.data[self.target_column])
            self.log("Encoded target column.", self.process_log)

            # Encode categorical columns
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != self.target_column:  # Avoid encoding target column again
                    self.data[col] = self.label_encoder.fit_transform(self.data[col])
            self.log("Encoded categorical columns.", self.process_log)

            # Balance the dataset
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            self.log("Balanced the dataset.", self.process_log)

            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=10)
            X_selected = selector.fit_transform(X_resampled, y_resampled)
            self.selected_features = X.columns[selector.get_support()]
            self.log("Selected top 10 features.", self.process_log)
            self.log(f"Selected features: {', '.join(self.selected_features)}", self.process_log)

            # Update self.data to include only selected features
            self.data_selected = pd.DataFrame(X_selected, columns=self.selected_features)
            self.data_selected[self.target_column] = y_resampled

            # Split data
            self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
                X_selected, y_resampled, test_size=0.2, random_state=42
            )
            self.log("Data processed successfully.", self.process_log)

            # Update prediction tab with selected feature names
            for i, feature in enumerate(self.selected_features):
                self.prediction_labels[i].config(text=f"{feature}:")
        except Exception as e:
            self.log(f"Error processing data: {e}", self.process_log)

    def save_clean_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data to save.")
            return

        # Ask user if they want to save only selected features
        save_option = messagebox.askyesno("Save Option", "Do you want to save only the selected features?")
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                if save_option:
                    self.data_selected.to_csv(file_path, index=False)  # Save selected features only
                else:
                    self.data.to_csv(file_path, index=False)  # Save all features
                messagebox.showinfo("Success", "Clean data saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {e}")

    def check_processing_steps(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data processed yet.")
            return

        # Create a new window to display processing steps
        steps_window = tk.Toplevel(self.root)
        steps_window.title("Processing Steps")
        steps_window.geometry("400x300")

        steps = [
            ("Load Data", self.data is not None),
            ("Drop Missing Values", self.data is not None and self.data.isnull().sum().sum() == 0),
            ("Encode Target Column", self.target_column in self.data.columns and self.data[self.target_column].dtype == 'int64'),
            ("Encode Categorical Columns", all(self.data[col].dtype == 'int64' for col in self.data.select_dtypes(include=['object']).columns)),
            ("Balance Dataset", self.train_data is not None and self.test_data is not None),
            ("Feature Selection", self.selected_features is not None),
            ("Split Data", self.train_data is not None and self.test_data is not None),
        ]

        for i, (step, status) in enumerate(steps):
            label = ttk.Label(steps_window, text=f"{step}: {'✔️' if status else '❌'}")
            label.pack(pady=5)

    def plot_data(self):
        selected_features = [self.feature_list.get(i) for i in self.feature_list.curselection()]
        if len(selected_features) == 0:
            messagebox.showwarning("Warning", "No features selected for plotting.")
            return
        if len(selected_features) > 3:
            messagebox.showwarning("Warning", "Please select a maximum of 3 features.")
            return

        try:
            plt.figure(figsize=(8, 5))
            if len(selected_features) == 1:
                sns.histplot(self.data[selected_features[0]], kde=True)
                plt.title(f"Distribution of {selected_features[0]}")
            elif len(selected_features) == 2:
                sns.scatterplot(data=self.data, x=selected_features[0], y=selected_features[1])
                plt.title(f"{selected_features[0]} vs {selected_features[1]}")
            elif len(selected_features) == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    self.data[selected_features[0]],
                    self.data[selected_features[1]],
                    self.data[selected_features[2]]
                )
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                ax.set_zlabel(selected_features[2])
                plt.title(f"{selected_features[0]}, {selected_features[1]}, {selected_features[2]}")
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot data: {e}")

    def train_model(self):
        if self.train_data is None or self.train_labels is None:
            self.log("No data to train on. Please process the data first.", self.model_log)
            return

        try:
            if self.model_combo.get() == "Deep Learning":
                self.model = keras.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(self.train_data.shape[1],)),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                self.model.fit(self.train_data, self.train_labels, epochs=50, batch_size=10, verbose=0)
                self.log("Deep Learning model trained successfully.", self.model_log)
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(random_state=42)
                self.model.fit(self.train_data, self.train_labels)
                self.log("Random Forest model trained successfully.", self.model_log)
        except Exception as e:
            self.log(f"Error training model: {e}", self.model_log)

    def test_model(self):
        if self.test_data is None or self.test_labels is None or self.model is None:
            self.log("No data to test on or model not trained.", self.model_log)
            return

        try:
            predictions = self.model.predict(self.test_data)
            if self.model_combo.get() == "Deep Learning":
                predictions = (predictions > 0.5).astype("int32")  # Convert predictions to classes (0 or 1)

            accuracy = accuracy_score(self.test_labels, predictions)
            precision = precision_score(self.test_labels, predictions, average='weighted')
            recall = recall_score(self.test_labels, predictions, average='weighted')

            self.log(f"Accuracy: {accuracy:.2f}", self.model_log)
            self.log(f"Precision: {precision:.2f}", self.model_log)
            self.log(f"Recall: {recall:.2f}", self.model_log)
        except Exception as e:
            self.log(f"Error testing model: {e}", self.model_log)

    def predict(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model trained. Please train a model first.")
            return

        try:
            # Clear previous prediction result
            self.prediction_result.config(text="")

            # Get input values from the user
            values = [float(entry.get()) if entry.get() else 0 for entry in self.prediction_entries]
            input_array = np.array(values).reshape(1, -1)

            # Make prediction
            prediction = self.model.predict(input_array)

            # Convert prediction to class label
            if self.model_combo.get() == "Deep Learning":
                prediction_class = (prediction > 0.5).astype("int32")[0][0]
            else:
                prediction_class = prediction[0]

            # Map prediction to target column labels
            if self.label_encoder:
                prediction_label = self.label_encoder.inverse_transform([prediction_class])[0]
            else:
                prediction_label = prediction_class

            # Display prediction result
            self.prediction_result.config(text=f"Prediction result: {prediction_label}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {e}")

    def new_prediction(self):
        # Clear all entry fields
        for entry in self.prediction_entries:
            entry.delete(0, tk.END)
        
        # Clear the prediction result
        self.prediction_result.config(text="Prediction result will appear here.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataApp(root)
    root.mainloop()