import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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
# sns.countplot(x="Survived",data=train)
# sns.countplot(x="Embarked",data=train)
# sns.countplot(x='Embarked',hue='Survived',data=train)


# sns.countplot(x='Embarked',hue='Survived',data=train)
# plt.show()


# sns.pairplot(train, vars=['Fare'],hue='Survived', diag_kind="kde",palette="coolwarm")
# plt.show()



fig = px.scatter(train, x="Age", y="Fare", color="Survived", title="Age vs Fare Scatter Plot")
fig.show()

