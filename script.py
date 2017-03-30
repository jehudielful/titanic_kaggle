import matplotlib.pyplot as plt
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

#train set processing 
count=0
sum_age=0

for i in range (0, len(train_data.Age)):
    if (pd.notnull(train_data.Age[i])):
        sum_age+=train_data.Age[i]
        count+=1
new_median_age=sum_age/count

train_data.Age[train_data.Age.isnull()]=new_median_age#train_data.Age.median()
MaxPassEmbarked = train_data.groupby('Embarked').count()['PassengerId']
train_data.Embarked[train_data.Embarked.isnull()]=MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#for item in range(1,len(train_data.Embarked)):
    #print (data.Embarked[item], end=" ") #print elements in line
label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
dicts['Sex'] = list (label.classes_)
data.Sex = label.transform(data.Sex) #заменяем значения из списка кодами закодированных элементов
label.fit(data.Embarked.drop_duplicates()) #задаем список значений для кодирования
dicts['Embarked'] = list (label.classes_)
data.Embarked = label.transform(data.Embarked) #заменяем значения из списка кодами закодированных элементов

#test set processing 
test_data.Age[test_data.Age.isnull()] = test_data.Age.mean()
test_data.Fare[test_data.Fare.isnull()]=test_data.Fare.median()
MaxPassEmbarked = test_data.groupby('Embarked').count()['PassengerId']
test_data.Embarked[test_data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = pd.DataFrame(test_data.PassengerId)
test = test_data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)
label.fit(dicts['Embarked'])
test.Embarked = label.transform(test.Embarked)

target = data.Survived
train = data.drop(['Survived'], axis=1) #из исходных данных убираем Id пассажира и флаг спасся он или нет
kfold = 5 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов

model_rfc = RandomForestClassifier(n_estimators = 70)
model_rfc.fit(train, target)
result.insert(1,'Survived', model_rfc.predict(test))

result.to_csv('C:/_WWWork/_git/titanic/data/result_test.csv', index=False)

#https://habrahabr.ru/post/202090/
#0.72727
#0.73206
