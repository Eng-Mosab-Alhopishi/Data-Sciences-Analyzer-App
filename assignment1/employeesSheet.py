import dask.dataframe as dd
import matplotlib.pyplot as plt

# Load the CSV file using Dask
df = dd.read_csv("./companies.csv")

# نسخ البيانات الأصلية لعرضها قبل المعالجة
df_original = df.copy()

name = df["name"]
salary = df["salary"]

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
            graphs_menu()
        else:
            print("Invalid choice")
        ch = int(input("\nEnter your required choice: "))

def DataFrame():
    print(df.compute())

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
    print(df["salary"].max().compute())

def Min():
    print("Minimum salary among employees:")
    print(df["salary"].min().compute())

def Am():
    print("Sum of salaries:")
    print(df["salary"].sum().compute())

def graphs_menu():
    print("-- GRAPHS MENU --")
    print("1. Line Graph (Before Cleaning)")
    print("2. Vertical Bar Graph (Before Cleaning)")
    print("3. Line Graph (After Cleaning)")
    print("4. Vertical Bar Graph (After Cleaning)")
    print("5. Exit")
    
    chg = int(input("Enter your required choice: "))
    
    while chg != 5:
        if chg == 1:
            line(df_original)
        elif chg == 2:
            barvertical(df_original)
        elif chg == 3:
            line(df)
        elif chg == 4:
            barvertical(df)
        else:
            print("Invalid choice")
        chg = int(input("\nEnter your required choice: "))

def line(data):
    plt.plot(data["name"].compute(), data["salary"].compute(), color="r", linewidth=7)
    plt.xlabel("NAME")
    plt.ylabel("SALARY")
    plt.title("EMPLOYEE SALARY SHEET")
    plt.show()

def barvertical(data):
    plt.bar(data["name"].compute(), data["salary"].compute(), linewidth=4)
    plt.xlabel("NAME")
    plt.ylabel("SALARY")
    plt.title("EMPLOYEE SALARY SHEET")
    plt.ylim(0, 20000)
    plt.xticks(rotation=90)
    plt.show()

# تنظيف البيانات
def clean_data():
    global df
    # حذف الصفوف التي تحتوي على قيم مفقودة
    df = df.dropna()

    # إزالة الصفوف التي تحتوي على قيم غير صحيحة
    df = df[df["salary"] > 0]

    # إزالة الصفوف المكررة
    df = df.drop_duplicates()

    # معالجة القيم المتطرفة
    q1 = df["salary"].quantile(0.25).compute()
    q3 = df["salary"].quantile(0.75).compute()
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df["salary"] >= lower_bound) & (df["salary"] <= upper_bound)]

    # معالجة الأخطاء الإملائية (بشكل مبسط)
    df["name"] = df["name"].str.strip().str.title()

    # معالجة التنسيق غير المتسق (بشكل مبسط)
    df["salary"] = df["salary"].astype(float)

# استدعاء دالة تنظيف البيانات
clean_data()

mainmenu()
