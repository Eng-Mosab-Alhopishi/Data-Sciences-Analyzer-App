import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Markdown log initialization
log = []

def log_step(step_number, description, output):
    """
    Logs each step with its description and output for the Markdown file.
    """
    log.append(f"### Step {step_number}: {description}\n\n```\n{output}\n```\n")

# Step 1: Load Data
step_number = 1
description = "Load Data"
data = pd.read_csv("WA.csv")
output = "Data loaded successfully."
log_step(step_number, description, output)

# Step 2: Display Basic Information
step_number += 1
description = "Display Basic Information"
output = f"First 5 rows:\n{data.head()}\n\nColumn Info:\n{data.info()}\n\nGeneral Statistics:\n{data.describe()}"
log_step(step_number, description, output)

# Step 3: Check for Missing Values
step_number += 1
description = "Check for Missing Values"
missing_values = data.isnull().sum()
output = f"Missing values:\n{missing_values}"
if missing_values.sum() == 0:
    output += "\nNo missing values detected."
else:
    output += "\nMissing values found. Please handle them."
log_step(step_number, description, output)

# Step 4: Drop Irrelevant Columns
# step_number += 1
# description = "Drop Irrelevant Columns"
# irrelevant_columns = ['EmployeeCount', 'Over18', 'StandardHours']
# data.drop(columns=irrelevant_columns, inplace=True)
# output = f"Irrelevant columns dropped: {irrelevant_columns}"
# log_step(step_number, description, output)



# Dropping irrelevant columns if they exist
irrelevant_columns = ['EmployeeCount', 'Over18', 'StandardHours']
columns_to_drop = [col for col in irrelevant_columns if col in data.columns]
data.drop(columns=columns_to_drop, inplace=True)

# Step 5: Convert Categorical Columns to Numeric (One-Hot Encoding)
step_number += 1
description = "Convert Categorical Columns to Numeric"
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
output = f"One-Hot Encoding performed on columns: {categorical_columns}"
log_step(step_number, description, output)

# Step 6: Check for Outliers or Strange Values
step_number += 1
description = "Check for Outliers or Strange Values"
output = f"General Statistics after processing:\n{data.describe()}"
if (data['MonthlyIncome'] <= 0).any():
    output += "\nFound invalid values in MonthlyIncome. Removing rows..."
    data = data[data['MonthlyIncome'] > 0]
    output += "\nInvalid values removed."
else:
    output += "\nNo invalid values detected in MonthlyIncome."
log_step(step_number, description, output)

# Step 7: Analyze Data
step_number += 1
description = "Analyze Data"
sns.boxplot(x='Attrition', y='Age', data=data)
plt.title("Attrition vs Age")
plt.savefig('attrition_vs_age.png')  # Save the plot
plt.close()
output = "Analysis plot saved as 'attrition_vs_age.png'."
log_step(step_number, description, output)

# Step 8: Split Data into Features and Target
step_number += 1
description = "Split Data into Features and Target"
X = data.drop(columns=['Attrition'])
y = data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
output = "Data split into training and test sets."
log_step(step_number, description, output)

# Step 9: Save Cleaned Data
step_number += 1
description = "Save Cleaned Data"
data.to_csv('cleaned_data.csv', index=False)
output = "Cleaned data saved successfully as 'cleaned_data.csv'."
log_step(step_number, description, output)

# Save the Markdown log
with open("data_processing_log.md", "w", encoding="utf-8") as log_file:
    log_file.write("# Data Processing Steps Log\n\n")
    log_file.write("\n".join(log))

print("All steps completed successfully. Log saved as 'data_processing_log.md'.")




















