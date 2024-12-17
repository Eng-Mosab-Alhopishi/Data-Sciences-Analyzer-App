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
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def min_salary(self):
        print("Minimum salary among employees:")
        print(self.df["salary"].min().compute())
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def max_salary(self):
        print("Maximum salary among employees:")
        print(self.df["salary"].max().compute())
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def sum_salaries(self):
        print("Sum of salaries:")
        print(self.df["salary"].sum().compute())
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def avg_salary_by_rating(self):
        data = self.df.compute()
        data.groupby("rating")["salary"].mean().plot(kind="bar")
        plt.xlabel("Rating")
        plt.ylabel("Average Salary")
        plt.title("Average Salary by Company Rating")
        plt.show()
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def salary_distribution_by_reviews(self):
        data = self.df.compute()
        data["reviews_group"] = (data["reviews"] // 50) * 50
        data.groupby("reviews_group")["salary"].mean().plot(kind="bar")
        plt.xlabel("Number of Reviews (Grouped)")
        plt.ylabel("Average Salary")
        plt.title("Average Salary by Number of Reviews (Grouped)")
        plt.show()
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def rating_distribution_by_reviews(self):
        data = self.df.compute()
        data.groupby("rating")["reviews"].mean().plot(kind="bar")
        plt.xlabel("Rating")
        plt.ylabel("Average Number of Reviews")
        plt.title("Average Number of Reviews by Company Rating")
        plt.show()
        input("Press any key to return to the main menu...")
        os.system('cls' if os.name == 'nt' else 'clear')

    def graphs_menu(self):
        print("-- GRAPHS MENU --")
        print("1. Average Salary by Company Rating")
        print("2. Average Salary by Number of Reviews (Grouped)")
        print("3. Average Number of Reviews by Company Rating")
        print("4. Exit")
        
        chg = int(input("Enter your required choice: "))
        
        while chg != 4:
            if chg == 1:
                self.avg_salary_by_rating()
            elif chg == 2:
                self.salary_distribution_by_reviews()
            elif chg == 3:
                self.rating_distribution_by_reviews()
            else:
                print("Invalid choice")
            chg = int(input("\nEnter your required choice: "))
            os.system('cls' if os.name == 'nt' else 'clear')

def mainmenu():
    processor = DataProcessor("./companies.csv")
    print("-- MAIN MENU --")
    print("1. Display DataFrame")
    print("2. Data Analysis")
    print("3. Visualizing the Data")
    print("4. Exit")
    
    ch = int(input("Enter your required choice: "))
    
    while ch != 4:
        if ch == 1:
            processor.display_dataframe()
        elif ch == 2:
            print("-- DATA ANALYSIS MENU --")
            print("1. Min Salary")
            print("2. Max Salary")
            print("3. Sum of Salaries")
            
            cha = int(input("Enter your required choice: "))
            
            while cha != 4:
                if cha == 1:
                    processor.min_salary()
                elif cha == 2:
                    processor.max_salary()
                elif cha == 3:
                    processor.sum_salaries()
                else:
                    print("Invalid choice")
                cha = int(input("\nEnter your required choice: "))
                os.system('cls' if os.name == 'nt' else 'clear')
        elif ch == 3:
            processor.graphs_menu()
        else:
            print("Invalid choice")
        ch = int(input("\nEnter your required choice: "))
        os.system('cls' if os.name == 'nt' else 'clear')

mainmenu()
