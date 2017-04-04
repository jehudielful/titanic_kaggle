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
#print(train_data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count')) #Pivot table (сводная таблица)
#pd.pivot_table(train_data,index=["Pclass"],values=["Fare"]).plot(kind="bar", stacked=True)
#print(train_data.PassengerId[train_data.Cabin.notnull()].count())
#plt.show()

#train set processing 
#----------------------Train Median Age Calculation--------------------------
median_age_sum=[0 for x in range(4)]
median_age_count=[0 for x in range(4)]
median_age=[0 for x in range(4)]

for i in range (0, len(train_data.Age)):
    if ((pd.notnull(train_data.Age[i])) and (train_data.Age[i]!=0)):
        if ("Master" in train_data.Name[i]):
            median_age_count[0]+=1
            median_age_sum[0]+=train_data.Age[i]
        elif ("Mrs" in train_data.Name[i]) or ("Countess" in train_data.Name[i]) or ("Jonkheer" in train_data.Name[i]):
            median_age_count[1]+=1
            median_age_sum[1]+=train_data.Age[i]
        elif ("Miss." in train_data.Name[i]):
            median_age_count[2]+=1
            median_age_sum[2]+=train_data.Age[i]
        else:
            median_age_count[3]+=1
            median_age_sum[3]+=train_data.Age[i]

for x in range(4):
    median_age[x]=median_age_sum[x]/median_age_count[x]
print (median_age)
for i in range (len(train_data.Age)):
    if (pd.isnull(train_data.Age[i])) or (train_data.Age[i]==0):
        if ("Master" in train_data.Name[i]):
            train_data.Age[i]=median_age[0]
        elif ("Mrs" in train_data.Name[i]) or ("Countess" in train_data.Name[i]) or ("Jonkheer" in train_data.Name[i]):
            train_data.Age[i]=median_age[1]
        elif ("Miss" in train_data.Name[i]):
            train_data.Age[i]=median_age[2]
        else:
            train_data.Age[i]=median_age[3]
#----------------------/Train Median Age Calculation--------------------------

#train_data.Age[train_data.Age.isnull()]=new_median_age #train_data.Age.median()
#MaxPassEmbarked = train_data.groupby('Embarked').count()['PassengerId']
#train_data.Embarked[train_data.Embarked.isnull()]=MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
data = train_data.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'],axis=1)
#for item in range(1,len(train_data.Embarked)):
    #print (data.Embarked[item], end=" ") #print elements in line
label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
dicts['Sex'] = list (label.classes_)
data.Sex = label.transform(data.Sex) #заменяем значения из списка кодами закодированных элементов
#label.fit(data.Embarked.drop_duplicates()) #задаем список значений для кодирования
#dicts['Embarked'] = list (label.classes_)
#data.Embarked = label.transform(data.Embarked) #заменяем значения из списка кодами закодированных элементов

#test set processing 
median_fare_sum=[0 for x in range(3)]
median_fare_count=[0 for x in range(3)]
median_fare=[0 for x in range(3)]
#matrix_fare = [[0 for x in range(3)] for y in range(len(train_data.Fare))] #двумерный массив 

#-------Train Median Fare Calculation-----------
for i in range (len(train_data.Fare)):
    if ((pd.notnull(train_data.Fare[i])) and (train_data.Fare[i]!=0)):
        if (train_data.Pclass[i]==1):
            median_fare_count[0]+=1
            median_fare_sum[0]+=train_data.Fare[i]
        elif (train_data.Pclass[i]==2):
            median_fare_count[1]+=1
            median_fare_sum[1]+=train_data.Fare[i]
        else:
            median_fare_count[2]+=1
            median_fare_sum[2]+=train_data.Fare[i]
for x in range(3):
    median_fare[x]=median_fare_sum[x]/median_fare_count[x]

for i in range (len(train_data.Fare)):
    if (pd.isnull(train_data.Fare[i])) or (train_data.Fare[i]==0):
        if (train_data.Pclass[i]==1):
           train_data.Fare[i]=median_fare[0]
        elif (train_data.Pclass[i]==2):
           train_data.Fare[i]=median_fare[1]
        else:
           train_data.Fare[i]=median_fare[2]
#-------/Train Median Fare Calculation-----------

#-------/Test Median Age Calculation-----------
for i in range (len(test_data.Age)):
    if (pd.isnull(test_data.Age[i])) or (test_data.Age[i]==0):
        if ("Master" in test_data.Name[i]):
            test_data.Age[i]=median_age[0]
        elif ("Mrs" in test_data.Name[i]) or ("Countess" in test_data.Name[i]) or ("Jonkheer" in test_data.Name[i]):
            test_data.Age[i]=median_age[1]
        elif ("Miss" in train_data.Name[i]):
            test_data.Age[i]=median_age[2]
        else:
            test_data.Age[i]=median_age[3]
#-------/Test Median Age Calculation-----------

for i in range (0, len(test_data.Fare)):
    if (pd.isnull(test_data.Fare[i])) or (test_data.Fare[i]==0):
        if (test_data.Pclass[i]==1):
           test_data.Fare[i]=median_fare[0]
        elif (test_data.Pclass[i]==2):
           test_data.Fare[i]=median_fare[1]
        else:
           test_data.Fare[i]=median_fare[2]
#test_data.Fare[test_data.Fare.isnull()]=new_median_fare #test_data.Fare[test_data.Fare.isnull()]=test_data.Fare.median()
#MaxPassEmbarked = test_data.groupby('Embarked').count()['PassengerId']
#test_data.Embarked[test_data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = pd.DataFrame(test_data.PassengerId)
test = test_data.drop(['Name','Ticket','Cabin','PassengerId', 'Embarked'],axis=1)
label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)
#label.fit(dicts['Embarked'])
#test.Embarked = label.transform(test.Embarked)

target = data.Survived
train = data.drop(['Survived'], axis=1) #из исходных данных убираем Id пассажира и флаг спасся он или нет

model_rfc = RandomForestClassifier(n_estimators = 95, max_features='auto', criterion='gini',max_depth=5)
model_rfc.fit(train, target)
result.insert(1,'Survived', model_rfc.predict(test))

result.to_csv('C:/_WWWork/_git/titanic/data/result_test.csv', index=False)

#https://habrahabr.ru/post/202090/
#0.72727
#0.73206
#OneHotEncoder
