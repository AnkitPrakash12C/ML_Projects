import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
# import plotly.express as px

train = pd.read_csv('D:\\Sem6\\ML-Projects\\train.csv')
test = pd.read_csv('D:\\Sem6\\ML-Projects\\train.csv')

# print(train.head(5))

# print(train.info())

# print(train.shape)

# print(train.isnull().sum())

train = train.drop(columns='Cabin',axis=1)
train.fillna({'Age': train['Age'].mean()}, inplace=True)
train.fillna({'Embarked':train['Embarked'].mode()[0]},inplace=True)

# print(train.isnull().sum())

# print(train['Embarked'].mode())
# print(train['Embarked'].mode()[0])

# print(train['Survived'].value_counts())

sns.set()
sns.countplot(x="Survived",data=train)
sns.countplot(x="Embarked",data=train)
sns.countplot(x='Embarked',hue='Survived',data=train)


sns.countplot(x='Embarked',hue='Survived',data=train)
# plt.show()


sns.pairplot(train, vars=['Fare'],hue='Survived', diag_kind="kde",palette="coolwarm")
# plt.show()



# fig = px.scatter(train, x="Age", y="Fare", color="Survived", title="Age vs Fare Scatter Plot")
# fig.show()

# print(train['Sex'].value_counts())
# print(train['Embarked'].value_counts())

train.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace=True)
# print(train.head())

x = train.drop(columns = ["PassengerId", "Name", "Ticket", "Survived"], axis=1)
y = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

# sc = StandardScaler()
# x_train_scaled =  sc.fit_transform(x_train)
# x_test_scaled = sc.transform(x_test)

model1_LR = LogisticRegression(max_iter=1000)
model1_LR.fit(x_train, y_train)

x_train_pred_1 = model1_LR.predict(x_train)
train_data_accuracy_1 = accuracy_score(y_train, x_train_pred_1)
# print(f"Accuracy score is: {train_data_accuracy_1 * 100:.2f}%")


model2_KNC = KNeighborsClassifier(n_neighbors=5)
model2_KNC.fit(x_train, y_train)

x_train_pred_2 = model2_KNC.predict(x_train)
train_data_accuracy_2 = accuracy_score(y_train, x_train_pred_2)
# print(f"Accuracy score is: {train_data_accuracy_2 * 100:.2f}%")

model3_DT = DecisionTreeClassifier()
model3_DT.fit(x_train, y_train)

x_train_pred_3 = model3_DT.predict(x_train)
train_data_accuracy_3 = accuracy_score(y_train, x_train_pred_3)
# print(f"Accuracy score is: {train_data_accuracy_3 * 100:.2f}%")

model4_RFC = RandomForestClassifier(n_estimators=100)
model4_RFC.fit(x_train, y_train)

x_train_pred_4 = model4_RFC.predict(x_train)
train_data_accuracy_4 = accuracy_score(y_train, x_train_pred_4)
# print(f"Accuracy score is: {train_data_accuracy_4 * 100:.2f}%")

model5_SVC = SVC(kernel='rbf', probability=True)
model5_SVC.fit(x_train, y_train)

x_train_pred_5 = model5_SVC.predict(x_train)
train_data_accuracy_5 = accuracy_score(y_train, x_train_pred_5)
# print(f"Accuracy score is: {train_data_accuracy_5 * 100:.2f}%")

model6_GNB = GaussianNB()
model6_GNB.fit(x_train, y_train)

x_train_pred_6 = model6_GNB.predict(x_train)
train_data_accuracy_6 = accuracy_score(y_train, x_train_pred_6)
# print(f"Accuracy score is: {train_data_accuracy_6 * 100:.2f}%")

model7_GBC = GradientBoostingClassifier()
model7_GBC.fit(x_train, y_train)

x_train_pred_7 = model7_GBC.predict(x_train)
train_data_accuracy_7 = accuracy_score(y_train, x_train_pred_7)
# print(f"Accuracy score is: {train_data_accuracy_7 * 100:.2f}%")

model8_XGB = XGBClassifier()
model8_XGB.fit(x_train, y_train)

x_train_pred_8 = model8_XGB.predict(x_train)
train_data_accuracy_8 = accuracy_score(y_train, x_train_pred_8)
# print(f"Accuracy score is: {train_data_accuracy_8 * 100:.2f}%")

model9_LGBM = LGBMClassifier()
model9_LGBM.fit(x_train, y_train)

x_train_pred_9 = model9_LGBM.predict(x_train)
train_data_accuracy_9 = accuracy_score(y_train, x_train_pred_9)
# print(f"Accuracy score is: {train_data_accuracy_9 * 100:.2f}%")

model10_CBC = CatBoostClassifier(verbose=0)
model10_CBC.fit(x_train, y_train)

x_train_pred_10 = model10_CBC.predict(x_train)
train_data_accuracy_10 = accuracy_score(y_train, x_train_pred_10)
# print(f"Accuracy score is: {train_data_accuracy_10 * 100:.2f}%")

model11_ABC = AdaBoostClassifier()
model11_ABC.fit(x_train, y_train)

x_train_pred_11 = model11_ABC.predict(x_train)
train_data_accuracy_11 = accuracy_score(y_train, x_train_pred_11)
# print(f"Accuracy score is: {train_data_accuracy_11 * 100:.2f}%")

model12_ETC = ExtraTreesClassifier(n_estimators=100)
model12_ETC.fit(x_train, y_train)

x_train_pred_12 = model12_ETC.predict(x_train)
train_data_accuracy_12 = accuracy_score(y_train, x_train_pred_12)
# print(f"Accuracy score is: {train_data_accuracy_12 * 100:.2f}%")
