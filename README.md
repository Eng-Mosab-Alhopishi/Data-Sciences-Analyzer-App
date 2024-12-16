# Employee Salary Analysis and Visualization

This project loads employee salary data, analyzes it, and visualizes it using Python and the Pandas and Matplotlib libraries.

## Requirements

- Python 3.x
- Pandas
- Matplotlib

<br>
<br>

# Installation

## Git Repository

To clone the repository and get started, use the following commands:

```bash
git clone https://github.com/abdulrahmanRadan/data-science.git
cd data-science
```

## Project Structure

```
data-science/
└── assignment1/
    ├── employeesSheet.py
    └── companies.csv
```

- after git clone :

## Create environment

1. Create and activate a virtual environment:

- in data-science folder

  ```sh
  python -m venv myenv
  ```

2. and start the env

   - On Windows

   ```sh
   myenv\Scripts\activate  # On Windows
   ```

   - On macOS or Linux

   ```sh
   source myenv/bin/activate  # On macOS or Linux
   ```

3. Install the required libraries:
   ```sh
    pip install pandas matplotlib
   ```

## Usage

1. Download the CSV file containing employee data into the `assignment1` folder.
2. Ensure the `employeesSheet.py` file is in the `assignment1` folder.
3. Run the code:
   ```sh
    cd assignment1
    python employeesSheet.py
   ```

## Code Description

- **employeesSheet.py**: Contains the code that loads, analyzes, and visualizes the data.
  - **mainmenu()**: Displays the main menu.
  - **DataFrame()**: Displays the DataFrame.
  - **DataAnalysis()**: Displays the data analysis menu.
    - **Max()**: Displays the maximum salary.
    - **Min()**: Displays the minimum salary.
    - **Am()**: Displays the sum of salaries.
  - **graphs()**: Displays the graphs menu.
    - **line()**: Plots a line graph of the data.
    - **barvertical()**: Plots a vertical bar graph of the data.

## Example Output

### Data Analysis Menu

- **Max Salary**: Displays the maximum salary among employees.
- **Min Salary**: Displays the minimum salary among employees.
- **Sum of Salaries**: Displays the total sum of salaries.

### Graphs Menu

- **Line Graph**: Plots a line graph of employee names vs. salaries.
- **Vertical Bar Graph**: Plots a vertical bar graph of employee names vs. salaries.

## Notes

- Ensure that the CSV file (`companies.csv`) is in the correct format with the appropriate column names (`name` and `salary`).
- The script provides a simple command-line interface to interact with the data and visualize it.

## Data Source

The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/iqmansingh/company-employee-dataset).

## Code Inspiration

The code for this project was inspired by the repository found [here](https://github.com/SANJAYSS-SRM-26/Employee-Salary-Analysis-and-Visualization-Python-/tree/main).

If you encounter any issues or need further assistance, feel free to reach out!
