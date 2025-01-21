import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def check_missing_values(self):
        missing_values = self.data.isnull().sum()
        return missing_values

    def drop_irrelevant_columns(self, columns):
        self.data.drop(columns=[col for col in columns if col in self.data.columns], inplace=True)

    def encode_categorical_columns(self, columns):
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)

    def remove_invalid_values(self, column, threshold):
        self.data = self.data[self.data[column] > threshold]

    def save_cleaned_data(self, file_path):
        self.data.to_csv(file_path, index=False)

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def create_boxplot(self, x, y, title, output_path):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x, y=y, data=self.data)
        plt.title(title)
        plt.savefig(output_path)
        plt.close()

    def create_countplot(self, x, hue, title, output_path):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=x, hue=hue, data=self.data)
        plt.title(title)
        plt.savefig(output_path)
        plt.close()

class ReportGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.content = ""

    def log_step(self, title, description):
        self.content += f"## {title}\n\n{description}\n\n"

    def log_visualization(self, title, image_path):
        self.content += f"### {title}\n![{title}]({image_path})\n\n"

    def save_report(self):
        with open(self.output_path, "w", encoding="utf-8") as file:
            file.write(self.content)

# Main Program
if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(project_dir, "WA.csv")
    cleaned_data_path = os.path.join(project_dir, "cleaned_data.csv")
    report_path = os.path.join(project_dir, "data_analysis_report.md")

    # Step 1: Load Data
    loader = DataLoader(raw_data_path)
    data = loader.load_data()

    # Step 2: Process Data
    processor = DataProcessor(data)
    missing_values = processor.check_missing_values()

    irrelevant_columns = ['EmployeeCount', 'Over18', 'StandardHours']
    processor.drop_irrelevant_columns(irrelevant_columns)

    categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    processor.encode_categorical_columns(categorical_columns)

    processor.remove_invalid_values('MonthlyIncome', 0)
    processor.save_cleaned_data(cleaned_data_path)

    # Step 3: Analyze Data
    analyzer = DataAnalyzer(processor.data)
    analyzer.create_boxplot('Attrition', 'Age', 'Attrition vs Age', os.path.join(project_dir, "attrition_vs_age.png"))
    analyzer.create_boxplot('Attrition', 'DistanceFromHome', 'Attrition vs DistanceFromHome', os.path.join(project_dir, "distance_from_home_vs_attrition.png"))
    analyzer.create_countplot('Gender', 'Attrition', 'Gender vs Attrition', os.path.join(project_dir, "gender_vs_attrition.png"))
    analyzer.create_countplot('MaritalStatus', 'Attrition', 'MaritalStatus vs Attrition', os.path.join(project_dir, "marital_status_vs_attrition.png"))

    # Step 4: Generate Report
    reporter = ReportGenerator(report_path)
    reporter.log_step("Missing Values", f"{missing_values}")
    reporter.log_visualization("Attrition vs Age", "attrition_vs_age.png")
    reporter.log_visualization("Attrition vs DistanceFromHome", "distance_from_home_vs_attrition.png")
    reporter.log_visualization("Gender vs Attrition", "gender_vs_attrition.png")
    reporter.log_visualization("MaritalStatus vs Attrition", "marital_status_vs_attrition.png")
    reporter.save_report()

    print("Process Completed Successfully!")
