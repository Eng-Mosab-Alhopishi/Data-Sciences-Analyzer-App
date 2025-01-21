# Data Processing Steps Log

### Step 1: Load Data

```
Data loaded successfully.
```

### Step 2: Display Basic Information

```
First 5 rows:
   Age Attrition  ... YearsSinceLastPromotion YearsWithCurrManager
0   41       Yes  ...                       0                    5
1   49        No  ...                       1                    7
2   37       Yes  ...                       0                    0
3   33        No  ...                       3                    0
4   27        No  ...                       2                    2

[5 rows x 24 columns]

Column Info:
None

General Statistics:
               Age  DistanceFromHome  ...  YearsSinceLastPromotion  YearsWithCurrManager
count  1470.000000       1470.000000  ...              1470.000000           1470.000000
mean     36.923810          9.192517  ...                 2.187755              4.123129
std       9.135373          8.106864  ...                 3.222430              3.568136
min      18.000000          1.000000  ...                 0.000000              0.000000
25%      30.000000          2.000000  ...                 0.000000              2.000000
50%      36.000000          7.000000  ...                 1.000000              3.000000
75%      43.000000         14.000000  ...                 3.000000              7.000000
max      60.000000         29.000000  ...                15.000000             17.000000

[8 rows x 16 columns]
```

### Step 3: Check for Missing Values

```
Missing values:
Age                         0
Attrition                   0
BusinessTravel              0
Department                  0
DistanceFromHome            0
Education                   0
EducationField              0
Gender                      0
JobLevel                    0
JobRole                     0
JobSatisfaction             0
MaritalStatus               0
MonthlyIncome               0
NumCompaniesWorked          0
OverTime                    0
PercentSalaryHike           0
PerformanceRating           0
RelationshipSatisfaction    0
TotalWorkingYears           0
TrainingTimesLastYear       0
WorkLifeBalance             0
YearsAtCompany              0
YearsSinceLastPromotion     0
YearsWithCurrManager        0
dtype: int64
No missing values detected.
```

### Step 4: Convert Categorical Columns to Numeric

```
One-Hot Encoding performed on columns: ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
```

### Step 5: Check for Outliers or Strange Values

```
General Statistics after processing:
               Age  DistanceFromHome  ...  YearsSinceLastPromotion  YearsWithCurrManager
count  1470.000000       1470.000000  ...              1470.000000           1470.000000
mean     36.923810          9.192517  ...                 2.187755              4.123129
std       9.135373          8.106864  ...                 3.222430              3.568136
min      18.000000          1.000000  ...                 0.000000              0.000000
25%      30.000000          2.000000  ...                 0.000000              2.000000
50%      36.000000          7.000000  ...                 1.000000              3.000000
75%      43.000000         14.000000  ...                 3.000000              7.000000
max      60.000000         29.000000  ...                15.000000             17.000000

[8 rows x 16 columns]
No invalid values detected in MonthlyIncome.
```

### Step 6: Analyze Data

```
Analysis plot saved as 'attrition_vs_age.png'.
```

### Step 7: Split Data into Features and Target

```
Data split into training and test sets.
```

### Step 8: Save Cleaned Data

```
Cleaned data saved successfully as 'cleaned_data.csv'.
```
