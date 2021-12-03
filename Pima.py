import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diabetes_data = pd.read_csv(r'C:\Users\imane\OneDrive\Desktop\Pima\diabetes.csv', sep =",", encoding = "utf-8")

diabetes_data.info()

diabetes_data.hist(bins=70, figsize=(20, 15))
plt.show()

sns.heatmap(diabetes_data.corr(), annot = True)
plt.show()

median_glucose = diabetes_data["Glucose"].median()
diabetes_data["Glucose"] =  diabetes_data["Glucose"].replace(to_replace = 0, value = median_glucose)

median_bp = diabetes_data["BloodPressure"].median()
diabetes_data["BloodPressure"] =  diabetes_data["BloodPressure"].replace(to_replace = 0, value = median_bp)

median_bmi = diabetes_data["BMI"].median()
diabetes_data["BMI"] =  diabetes_data["BMI"].replace(to_replace = 0, value = median_bmi)

median_insulin = diabetes_data["Insulin"].median()
diabetes_data["Insulin"] =  diabetes_data["Insulin"].replace(to_replace = 0, value = median_insulin)

median_st = diabetes_data["SkinThickness"].median()
diabetes_data["SkinThickness"] =  diabetes_data["SkinThickness"].replace(to_replace = 0, value = median_st)

diabetes_data.hist(bins=70, figsize=(20, 15))
plt.show()
sns.heatmap(diabetes_data.corr(), annot = True)
plt.show()
from sklearn.model_selection import train_test_split


train_set, test_set = train_test_split(diabetes_data, test_size = 0.2,random_state = 42)

train_set_labels = train_set["Outcome"].copy()
train_set = train_set.drop("Outcome", axis = 1)
test_set_labels = test_set["Outcome"].copy()
test_set = test_set.drop("Outcome", axis = 1)

from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(train_set)
scaled_train_set = scaler.transform(train_set)
scaled_test_set = scaler.transform(test_set)

scaled_df = pd.DataFrame(data = scaled_train_set)
print(scaled_df.head())