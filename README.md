# Company Salary Analysis and Visualization

This project loads company salary data, analyzes it, and visualizes it using Python, Dask, and Matplotlib libraries.

## Requirements

- Python 3.x
- Dask
- Matplotlib

---

## Installation

### Git Repository

To clone the repository and get started, use the following commands:

```bash
git clone https://github.com/abdulrahmanRadan/data-science.git
cd data-science
```

### Project Structure

```
data-science/
└── assignment1/
    ├── employeesSheet.py
    └── companies.csv
```

---

### Create Environment

1. Create and activate a virtual environment:

- In the `data-science` folder:

  ```sh
  python -m venv myenv
  ```

2. Activate the environment:

   - On Windows:

   ```sh
   myenv\Scripts\activate
   ```

   - On macOS or Linux:

   ```sh
   source myenv/bin/activate
   ```

3. Install the required libraries:

   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

1. Ensure the `companies.csv` file is in the `assignment1` folder and has the required columns: `name`, `salary`, `rating`, and `reviews`.
2. Run the program:

   ```sh
   cd final
   python datap.py
   ```

3. Follow the on-screen prompts to interact with the program.

---

## Code Description

- **employeesSheet.py**: Contains the code for loading, cleaning, analyzing, and visualizing the company salary data.

### Main Menu

1. **Display DataFrame**: Displays the cleaned DataFrame.
2. **Data Analysis Menu**:
   - **Min Salary**: Displays the minimum salary.
   - **Max Salary**: Displays the maximum salary.
   - **Sum of Salaries**: Displays the total sum of salaries.
3. **Graphs Menu**:

   - **Average Salary by Company Rating**: Visualizes the average salary grouped by company rating.
   - **Average Salary by Number of Reviews (Grouped)**: Visualizes the average salary grouped by the number of reviews.
   - **Average Number of Reviews by Company Rating**: Visualizes the average number of reviews for each company rating.

4. **Exit**: Exits the program.

### Data Cleaning

- Removed rows with missing values.
- Filtered out rows where salary is less than or equal to zero.
- Removed salary outliers using the Interquartile Range (IQR) method.
- Standardized employee names by trimming spaces and ensuring proper title case.
- Ensured salary values are treated as floats for analysis.

---

## Example Output

### Data Analysis Menu

- **Min Salary**: Displays the lowest salary value.
- **Max Salary**: Displays the highest salary value.
- **Sum of Salaries**: Displays the total sum of all salaries in the dataset.

### Graphs Menu

1. **Average Salary by Company Rating**:

   - Bar chart showing average salary per company rating.

2. **Average Salary by Number of Reviews (Grouped)**:

   - Bar chart showing average salary grouped by review count ranges (e.g., 0-50, 51-100).

3. **Average Number of Reviews by Company Rating**:
   - Bar chart showing the average number of reviews for each rating.

---

## Assignment1

### Tasks Completed

1. **Loading Data**: Loaded the company salary data from a CSV file using **Dask**.
2. **Cleaning Noisy Data**:
   - Removed rows with missing or invalid values.
   - Filtered out rows with negative or zero salaries.
   - Handled outliers using the Interquartile Range (IQR) method.
   - Standardized and formatted names.
3. **Data Analysis**:
   - Calculated the minimum salary.
   - Calculated the maximum salary.
   - Calculated the total sum of salaries.
4. **Visualization**:
   - Plotted average salary by company rating.
   - Plotted average salary grouped by number of reviews.
   - Plotted average number of reviews by company rating.

---

## Notes

- The `companies.csv` file should have the following columns:
  - `name`: Name of the company or employee.
  - `salary`: Salary of the employee (numeric).
  - `rating`: Rating of the company.
  - `reviews`: Number of reviews for the company.
- Missing or invalid rows are cleaned as part of the data preparation process.

---

## Data Source

The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/iqmansingh/company-employee-dataset).

---

If you encounter any issues or need further assistance, feel free to reach out!

---
