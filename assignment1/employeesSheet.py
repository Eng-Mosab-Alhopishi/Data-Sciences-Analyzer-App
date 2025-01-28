import os
import dask.dataframe as dd
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, file_path):
        self.df = dd.read_csv(file_path)
        self.df_original = self.df.copy()
        self.clean_data()

    def clean_data(self):
        self.df = self.df.dropna()
        self.df = self.df[self.df["salary"] > 0]
        q1 = self.df["salary"].quantile(0.25).compute()
        q3 = self.df["salary"].quantile(0.75).compute()
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        self.df = self.df[(self.df["salary"] >= lower_bound) & (self.df["salary"] <= upper_bound)]
        self.df["name"] = self.df["name"].str.strip().str.title()
        self.df["salary"] = self.df["salary"].astype(float)

    def display_dataframe(self):
        print(self.df.compute())

    def min_salary(self):
        return self.df["salary"].min().compute()

    def max_salary(self):
        return self.df["salary"].max().compute()

    def sum_salaries(self):
        return self.df["salary"].sum().compute()

    def avg_salary_by_rating(self):
        data = self.df.compute()
        data.groupby("rating")["salary"].mean().plot(kind="bar")
        plt.xlabel("Rating")
        plt.ylabel("Average Salary")
        plt.title("Average Salary by Company Rating")
        plt.show()

    def salary_distribution_by_reviews(self):
        data = self.df.compute()
        data["reviews_group"] = (data["reviews"] // 50) * 50
        data.groupby("reviews_group")["salary"].mean().plot(kind="bar")
        plt.xlabel("Number of Reviews (Grouped)")
        plt.ylabel("Average Salary")
        plt.title("Average Salary by Number of Reviews (Grouped)")
        plt.show()

    def rating_distribution_by_reviews(self):
        data = self.df.compute()
        data.groupby("rating")["reviews"].mean().plot(kind="bar")
        plt.xlabel("Rating")
        plt.ylabel("Average Number of Reviews")
        plt.title("Average Number of Reviews by Company Rating")
        plt.show()

def main_menu():
    print("-- MAIN MENU --")
    print("1. Display DataFrame")
    print("2. Data Analysis")
    print("3. Visualizing the Data")
    print("4. Exit")
    choice = input("Enter your required choice: ")
    return choice

def data_analysis_menu(processor):
    while True:
        print("-- DATA ANALYSIS MENU --")
        print("1. Min Salary")
        print("2. Max Salary")
        print("3. Sum of Salaries")
        print("4. Back to Main Menu")
        choice = input("Enter your required choice: ")
        if choice == "1":
            print(f"Min Salary: {processor.min_salary()}")
        elif choice == "2":
            print(f"Max Salary: {processor.max_salary()}")
        elif choice == "3":
            print(f"Sum of Salaries: {processor.sum_salaries()}")
        elif choice == "4":
            break
        else:
            print("Invalid input. Please try again.")

def graphs_menu(processor):
    while True:
        print("-- GRAPHS MENU --")
        print("1. Average Salary by Company Rating")
        print("2. Average Salary by Number of Reviews (Grouped)")
        print("3. Average Number of Reviews by Company Rating")
        print("4. Back to Main Menu")
        choice = input("Enter your required choice: ")
        if choice == "1":
            processor.avg_salary_by_rating()
        elif choice == "2":
            processor.salary_distribution_by_reviews()
        elif choice == "3":
            processor.rating_distribution_by_reviews()
        elif choice == "4":
            break
        else:
            print("Invalid input. Please try again.")

def main():
    processor = DataProcessor("./companies.csv")
    while True:
        choice = main_menu()
        if choice == "1":
            processor.display_dataframe()
        elif choice == "2":
            data_analysis_menu(processor)
        elif choice == "3":
            graphs_menu(processor)
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
