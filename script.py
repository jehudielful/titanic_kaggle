import matplotlib.pyplot as plt
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file
import math
from sklearn.preprocessing import LabelEncoder

#Load data
train_data = pd.read_csv('C:/_WWWork/_git/titanic/data/train.csv')
test_data = pd.read_csv('C:/_WWWork/_git/titanic/data/test.csv')
all_data = pd.concat([train_data, test_data]) 

#Important data
#print(train_data.head())
#print(train_data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count')) #Pivot table (сводная таблица)
#pd.pivot_table(train_data,index=["Pclass"],values=["Survived"]).plot(kind="bar", stacked=True)
#print(train_data.PassengerId[train_data.Cabin.notnull()].count())
#plt.show()
medianAge = train_data.Age.median()
MaxPassEmbarked = train_data.groupby('Embarked').count()['PassengerId']
MaxPassEmbarked = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
for i in range (0, len(train_data.Age)):
    if (math.isnan(train_data.Age[i])):
        train_data.Age[i]=medianAge
    if (train_data.Embarked[i]==""):
        train_data.Embarked[i]=MaxPassEmbarked
data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
dicts['Sex'] = list (label.classes_)
data.Sex = label.transform(data.Sex) #заменяем значения из списка кодами закодированных элементов

label.fit(data.Embarked.drop_duplicates()) #задаем список значений для кодирования
dicts['Embarked'] = list (label.classes_)
data.Embarked = label.transform(data.Embarked) #заменяем значения из списка кодами закодированных элементов

#print (data.head())
#https://habrahabr.ru/post/202090/
