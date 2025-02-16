import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


