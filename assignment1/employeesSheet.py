import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("./companies.csv")
sample_df = df.sample(n=100)

name = sample_df["name"]
salary = sample_df["salary"]

def mainmenu():
    print("-- MAIN MENU --")
    print("1. Display DataFrame")
    print("2. Data Analysis")
    print("3. Visualizing the Data")
    print("4. Exit")
    
    ch = int(input("Enter your required choice: "))
    
    while ch != 4:
        if ch == 1:
            DataFrame()
        elif ch == 2:
            DataAnalysis()
        elif ch == 3:
            graphs()
        else:
            print("Invalid choice")
        ch = int(input("\nEnter your required choice: "))

def DataFrame():
    print(sample_df)

def DataAnalysis():
    print("-- DATA ANALYSIS MENU --")
    print("1. Min Salary")
    print("2. Max Salary")
    print("3. Sum of Salaries")
    
    cha = int(input("Enter your required choice: "))
    
    while cha != 4:
        if cha == 1:
            Min()
        elif cha == 2:
            Max()
        elif cha == 3:
            Am()
        else:
            print("Invalid choice")
        cha = int(input("\nEnter your required choice: "))

def Max():
    print("Maximum salary among employees:")
    print(sample_df["salary"].max())

def Min():
    print("Minimum salary among employees:")
    print(sample_df["salary"].min())

def Am():
    print("Sum of salaries:")
    print(sample_df["salary"].sum())

def graphs():
    print("-- GRAPHS --")
    print("1. Line Graph")
    print("2. Vertical Bar Graph")
    print("3. Exit")
    
    chg = int(input("Enter your required choice: "))
    
    while chg != 3:
        if chg == 1:
            line()
        elif chg == 2:
            barvertical()
        else:
            print("Invalid choice")
        chg = int(input("\nEnter your required choice: "))

def line():
    plt.plot(name, salary, color="r", linewidth=7)
    plt.xlabel("NAME")
    plt.ylabel("SALARY")
    plt.title("EMPLOYEE SALARY SHEET")
    plt.show()

def barvertical():
    plt.bar(name, salary, linewidth=4)
    plt.xlabel("NAME")
    plt.ylabel("SALARY")
    plt.title("EMPLOYEE SALARY SHEET")
    plt.xticks(rotation=90)
    plt.show()


mainmenu()
