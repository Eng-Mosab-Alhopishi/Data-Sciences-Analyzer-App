# Data Analysis Report
This report summarizes the data analysis steps and visualizations for the dataset.

## 1. Data Overview
The first 5 rows of the data:
```
   Age Attrition  ... YearsSinceLastPromotion YearsWithCurrManager
0   41       Yes  ...                       0                    5
1   49        No  ...                       1                    7
2   37       Yes  ...                       0                    0
3   33        No  ...                       3                    0
4   27        No  ...                       2                    2

[5 rows x 24 columns]
```
## 2. Information about Columns
```
None
```
## 3. Descriptive Statistics
```
               Age  ...  YearsWithCurrManager
count  1470.000000  ...           1470.000000
mean     36.923810  ...              4.123129
std       9.135373  ...              3.568136
min      18.000000  ...              0.000000
25%      30.000000  ...              2.000000
50%      36.000000  ...              3.000000
75%      43.000000  ...              7.000000
max      60.000000  ...             17.000000

[8 rows x 17 columns]
```
## 4. MaritalStatus Column Transformation
The 'MaritalStatus' column was transformed from categorical values to numeric values.
Mapping: 'Single' -> 0, 'Married' -> 1, 'Divorced' -> 2.
First 5 rows after transformation:
```
   Age Attrition  ... YearsSinceLastPromotion YearsWithCurrManager
0   41       Yes  ...                       0                    5
1   49        No  ...                       1                    7
2   37       Yes  ...                       0                    0
3   33        No  ...                       3                    0
4   27        No  ...                       2                    2

[5 rows x 24 columns]
```
## 5. Visualizations
### Categorical Columns Distribution
Distribution of Attrition:
```
Attrition
No     1233
Yes     237
Name: count, dtype: int64
```
Distribution of BusinessTravel:
```
BusinessTravel
Travel_Rarely        1043
Travel_Frequently     277
Non-Travel            150
Name: count, dtype: int64
```
Distribution of Department:
```
Department
Research & Development    961
Sales                     446
Human Resources            63
Name: count, dtype: int64
```
Distribution of EducationField:
```
EducationField
Life Sciences       606
Medical             464
Marketing           159
Technical Degree    132
Other                82
Human Resources      27
Name: count, dtype: int64
```
Distribution of Gender:
```
Gender
Male      882
Female    588
Name: count, dtype: int64
```
Distribution of JobRole:
```
JobRole
Sales Executive              326
Research Scientist           292
Laboratory Technician        259
Manufacturing Director       145
Healthcare Representative    131
Manager                      102
Sales Representative          83
Research Director             80
Human Resources               52
Name: count, dtype: int64
```
Distribution of OverTime:
```
OverTime
No     1054
Yes     416
Name: count, dtype: int64
```
### Numerical Columns Distribution
Distribution of Age:
```
count    1470.000000
mean       36.923810
std         9.135373
min        18.000000
25%        30.000000
50%        36.000000
75%        43.000000
max        60.000000
Name: Age, dtype: float64
```
Distribution of DistanceFromHome:
```
count    1470.000000
mean        9.192517
std         8.106864
min         1.000000
25%         2.000000
50%         7.000000
75%        14.000000
max        29.000000
Name: DistanceFromHome, dtype: float64
```
Distribution of Education:
```
count    1470.000000
mean        2.912925
std         1.024165
min         1.000000
25%         2.000000
50%         3.000000
75%         4.000000
max         5.000000
Name: Education, dtype: float64
```
Distribution of JobLevel:
```
count    1470.000000
mean        2.063946
std         1.106940
min         1.000000
25%         1.000000
50%         2.000000
75%         3.000000
max         5.000000
Name: JobLevel, dtype: float64
```
Distribution of JobSatisfaction:
```
count    1470.000000
mean        2.728571
std         1.102846
min         1.000000
25%         2.000000
50%         3.000000
75%         4.000000
max         4.000000
Name: JobSatisfaction, dtype: float64
```
Distribution of MaritalStatus:
```
count    1470.000000
mean        0.902721
std         0.730121
min         0.000000
25%         0.000000
50%         1.000000
75%         1.000000
max         2.000000
Name: MaritalStatus, dtype: float64
```
Distribution of MonthlyIncome:
```
count     1470.000000
mean      6502.931293
std       4707.956783
min       1009.000000
25%       2911.000000
50%       4919.000000
75%       8379.000000
max      19999.000000
Name: MonthlyIncome, dtype: float64
```
Distribution of NumCompaniesWorked:
```
count    1470.000000
mean        2.693197
std         2.498009
min         0.000000
25%         1.000000
50%         2.000000
75%         4.000000
max         9.000000
Name: NumCompaniesWorked, dtype: float64
```
Distribution of PercentSalaryHike:
```
count    1470.000000
mean       15.209524
std         3.659938
min        11.000000
25%        12.000000
50%        14.000000
75%        18.000000
max        25.000000
Name: PercentSalaryHike, dtype: float64
```
Distribution of PerformanceRating:
```
count    1470.000000
mean        3.153741
std         0.360824
min         3.000000
25%         3.000000
50%         3.000000
75%         3.000000
max         4.000000
Name: PerformanceRating, dtype: float64
```
Distribution of RelationshipSatisfaction:
```
count    1470.000000
mean        2.712245
std         1.081209
min         1.000000
25%         2.000000
50%         3.000000
75%         4.000000
max         4.000000
Name: RelationshipSatisfaction, dtype: float64
```
Distribution of TotalWorkingYears:
```
count    1470.000000
mean       11.279592
std         7.780782
min         0.000000
25%         6.000000
50%        10.000000
75%        15.000000
max        40.000000
Name: TotalWorkingYears, dtype: float64
```
Distribution of TrainingTimesLastYear:
```
count    1470.000000
mean        2.799320
std         1.289271
min         0.000000
25%         2.000000
50%         3.000000
75%         3.000000
max         6.000000
Name: TrainingTimesLastYear, dtype: float64
```
Distribution of WorkLifeBalance:
```
count    1470.000000
mean        2.761224
std         0.706476
min         1.000000
25%         2.000000
50%         3.000000
75%         3.000000
max         4.000000
Name: WorkLifeBalance, dtype: float64
```
Distribution of YearsAtCompany:
```
count    1470.000000
mean        7.008163
std         6.126525
min         0.000000
25%         3.000000
50%         5.000000
75%         9.000000
max        40.000000
Name: YearsAtCompany, dtype: float64
```
Distribution of YearsSinceLastPromotion:
```
count    1470.000000
mean        2.187755
std         3.222430
min         0.000000
25%         0.000000
50%         1.000000
75%         3.000000
max        15.000000
Name: YearsSinceLastPromotion, dtype: float64
```
Distribution of YearsWithCurrManager:
```
count    1470.000000
mean        4.123129
std         3.568136
min         0.000000
25%         2.000000
50%         3.000000
75%         7.000000
max        17.000000
Name: YearsWithCurrManager, dtype: float64
```
### Attrition vs MaritalStatus
A plot showing the distribution of Attrition based on MaritalStatus.
### Attrition vs Numerical Columns
Analysis of Age against Attrition.
Analysis of DistanceFromHome against Attrition.
Analysis of Education against Attrition.
Analysis of JobLevel against Attrition.
Analysis of JobSatisfaction against Attrition.
Analysis of MaritalStatus against Attrition.
Analysis of MonthlyIncome against Attrition.
Analysis of NumCompaniesWorked against Attrition.
Analysis of PercentSalaryHike against Attrition.
Analysis of PerformanceRating against Attrition.
Analysis of RelationshipSatisfaction against Attrition.
Analysis of TotalWorkingYears against Attrition.
Analysis of TrainingTimesLastYear against Attrition.
Analysis of WorkLifeBalance against Attrition.
Analysis of YearsAtCompany against Attrition.
Analysis of YearsSinceLastPromotion against Attrition.
Analysis of YearsWithCurrManager against Attrition.
## 6. Correlation Analysis
The correlation heatmap shows the correlation between numerical columns.
```
                               Age  ...  YearsWithCurrManager
Age                       1.000000  ...              0.202089
DistanceFromHome         -0.001686  ...              0.014406
Education                 0.208034  ...              0.069065
JobLevel                  0.509604  ...              0.375281
JobSatisfaction          -0.004892  ...             -0.027656
MaritalStatus             0.095029  ...              0.038570
MonthlyIncome             0.497855  ...              0.344079
NumCompaniesWorked        0.299635  ...             -0.110319
PercentSalaryHike         0.003634  ...             -0.011985
PerformanceRating         0.001904  ...              0.022827
RelationshipSatisfaction  0.053535  ...             -0.000867
TotalWorkingYears         0.680381  ...              0.459188
TrainingTimesLastYear    -0.019621  ...             -0.004096
WorkLifeBalance          -0.021490  ...              0.002759
YearsAtCompany            0.311309  ...              0.769212
YearsSinceLastPromotion   0.216513  ...              0.510224
YearsWithCurrManager      0.202089  ...              1.000000

[17 rows x 17 columns]
```
