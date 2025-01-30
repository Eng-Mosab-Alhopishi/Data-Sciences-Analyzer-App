import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. إنشاء البيانات الافتراضية
np.random.seed(42)  # لضمان تكرار النتائج
n_records = 10000
data = {
    "Age": np.random.randint(18, 65, n_records),
    "Salary": np.random.randint(3000, 15000, n_records),
    "Experience": np.random.randint(1, 30, n_records),
    "Department": np.random.choice(["HR", "IT", "Finance", "Marketing"], n_records)
}
df = pd.DataFrame(data)

# 2. إضافة الضوضاء
# إضافة القيم الشاذة
df.loc[np.random.choice(df.index, 50), "Salary"] *= 10
# إضافة القيم المفقودة
df.loc[np.random.choice(df.index, 100), "Experience"] = np.nan

# 3. التصور
plt.figure(figsize=(12, 6))
sns.boxplot(x="Department", y="Salary", data=df)
plt.title("Salary Distribution by Department (with Noise)")
plt.show(block=False)  # عرض الرسم بدون تعطيل

# التوزيع العمري
plt.figure(figsize=(8, 6))
sns.histplot(df["Age"], kde=True, bins=30)
plt.title("Age Distribution")
plt.show(block=False)  # عرض الرسم بدون تعطيل

# لإبقاء النوافذ مفتوحة
plt.show()  # يبقي النوافذ مرئية حتى يتم إغلاقها يدويًا
