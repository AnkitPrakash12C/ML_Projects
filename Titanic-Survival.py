import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
sns.countplot(x='Embarked',hue='Survived',data=train)
plt.show()

