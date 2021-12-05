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


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from  sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn import model_selection

models = []
models.append(("LogReg", LogisticRegression()))
models.append(("RanFor", RandomForestClassifier()))
models.append(("DecTree", DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')))
models.append(("KNN",KNeighborsClassifier()))
models.append(("SVC",SVC()))

seed = 7
results = []
names = []
X = scaled_train_set
Y = train_set_labels


for name, model in models:
    kfolds = model_selection.KFold(n_splits=10)
    cv_results  = model_selection.cross_val_score(model, X, Y, cv=kfolds, scoring="accuracy")
    results.append(cv_results)
    names.append(name)

    outcome = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(outcome)

figure = plt.figure()
figure.suptitle("Algorithm performance")
ax = figure.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()





