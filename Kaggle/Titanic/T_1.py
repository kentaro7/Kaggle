import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import seaborn as sns 
import random
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("."))
['gender_submission.csv','train.csv','test.csv']

train = pd.read_csv("/Users/kanari/Kaggle/Titanic/train.csv")
test = pd.read_csv("/Users/kanari/Kaggle/Titanic/test.csv")

print("\n")
print(train.info())
print("*"*30)
print(test.info())
print("\n")

all_data = pd.concat([train,test],axis=0,ignore_index=True)
all_data.drop("Survived",inplace=True,axis=1)
print(all_data.info())

###データ整理

####Survived

print("\n")
print("About Survived")
print("*"*30)
print(train["Survived"].value_counts())

####Pclass
print("\n")
print("About Pclass")
print("*"*30)
print(all_data["Pclass"].value_counts())

####Name
print("\n")
print("About Name")
print("*"*30)
all_data["Title"] = [i.split('.')[0] for i in all_data["Name"]]
all_data["Name"] = [i.split(',')[1] for i in all_data["Title"]]
print(all_data["Name"].value_counts())

##男を1,女を2,他を3,masterを4,Missを5
all_data["Name"] = [i.replace("Mr","1") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Miss","2") for i in all_data["Name"]]
all_data["Name"] = [i.replace('Mrs',"2") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Master","4") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Dr","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Rev","1") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Col","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Mlle","2") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Ms","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Major","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Dona","2") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Don","1") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Capt","1") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Lady","2") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Jonkheer","2") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Mme","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace("the Countess","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace("Sir","3") for i in all_data["Name"]]
all_data["Name"] = [i.replace('1s',"2") for i in all_data["Name"]]

print(all_data["Name"].value_counts())
all_data = pd.get_dummies(data = all_data,columns=["Name"])
all_data.drop("Title",axis = 1,inplace=True)

###Sex
print("\n")
print("About Sex")
print("*"*30)
#print(all_data["Sex"].isnull().sum())
print(all_data["Sex"].value_counts())
all_data = pd.get_dummies(data=all_data,columns=["Sex"])

###Age
print("\n")
print("About Age")
print("*"*30)
print("sum of null : "+str(all_data["Age"].isnull().sum()))
##Null値を当てはめる

###SibSp -> タイタニックに乗っていない夫婦、兄弟の数
print("\n")
print("About SibSp")
print("*"*30)
#print(all_data["SibSp"].isnull().sum())
print(all_data["SibSp"].value_counts())

###Parch　-> タイタニックに乗っていない息子、娘、親の数
print("\n")
print("About Parch")
print("*"*30)
#print(all_data["Parch"].isnull().sum())

###Ticket
print("\n")
print("About Ticket")
print("*"*30)
#print(all_data["Ticket"].isnull().sum())
###文字などの種類を特定し、数字だけにする。

###Fare
print("\n")
print("About Fare")
print("*"*30)
#print(all_data[all_data["Fare"].isnull()]==True)
##PassengerId = 1043　のFareを当てはめる

###Cabin
print("\n")
print("About Cabin")
print("*"*30)
print(all_data["Cabin"].isnull().sum())
##Null値を当てはめて、Cabinを統一する。

###Embarked
print("\n")
print("About Embarked")
print("*"*30)
print(all_data["Embarked"].isnull().sum())

#####NULL値埋める

###Age
print("\n")
print("Age data")
print("*"*30)

#ax = plt.hist(all_data.Age)
#plt.savefig("Age")
#print("Max : "+str(np.max(all_data["Age"])))
#print("Min : "+str(np.min(all_data["Age"])))
#print("Mean : "+str(np.mean(all_data["Age"])))
#print("\n")

#sns.kdeplot(train.Age[train["Survived"]==0],shade=True,label="Not")
#sns.kdeplot(train.Age[train["Survived"]==1],shade=True,label="Yes")
#plt.savefig("Age and Survived")

#sns.kdeplot(train.Age[train["Sex"]=="male"],shade=True,label="male")
#sns.kdeplot(train.Age[train["Sex"]=="female"],shade=True,label="female")
#plt.savefig("Age and Sex")

##name is master ---> 1~10year

#print("Masterの平均 : "+str(all_data.Age[all_data["Name_ 4"]==1].mean()))

#all_data.iloc[5,0] = 5.0
#for i in [1308,65,67,159,176,709,1135,1230,1235]:
    
 #   all_data.iloc[i,0] = 5.4  

#print(all_data.Age[all_data["Name_ 4"]==1].isnull().sum())

##name is women
#print("\n")
#print("womenの平均　: "+str(all_data.Age[all_data["Name_ 2"]==1].mean()))
#print("womenの中央値　: "+str(all_data.Age[all_data["Name_ 2"]==1].median()))
#print("womenの最大値 : "+str(all_data.Age[all_data["Name_ 2"]==1].max()))
#print("womenの最小値 : "+str(all_data.Age[all_data["Name_ 2"]==1].min()))

#print(all_data.Age[all_data["Name_ 2"]==1].isnull())

#for i in [19,31,140,166,186,256,334,347,367,375,415,431,457,533,578,669,849,913,924,956,1023,1059,1090,1116,1140,1256,1273]:
    
 #   all_data.iloc[i,0] = 36.0

#print(all_data.Age[all_data["Name_ 2"]==1].isnull())

##name is Mr.---> 

##name is その他
#print("\n")
#print("その他の平均　: "+str(all_data.Age[all_data["Name_ 3"]==1].mean()))
#print("その他の中央値　: "+str(all_data.Age[all_data["Name_ 3"]==1].median()))
#print("その他の最大値 : "+str(all_data.Age[all_data["Name_ 3"]==1].max()))
#print("その他の最小値 : "+str(all_data.Age[all_data["Name_ 3"]==1].min()))
#print(all_data.Age[all_data["Name_ 3"]==1].isnull())
#for i in [766,979]:
    
#    all_data.iloc[i,0] = 46.0

##name is Miss
#print("\n")
#print("Missの平均　: "+str(all_data.Age[all_data["Name_ 5"]==1].mean()))
#print("Missの中央値　: "+str(all_data.Age[all_data["Name_ 5"]==1].median()))
#print("Missの最大値 : "+str(all_data.Age[all_data["Name_ 5"]==1].max()))
#print("Missの最小値 : "+str(all_data.Age[all_data["Name_ 5"]==1].min()))
#print(all_data.Age[all_data["Name_ 5"]==1].isnull())

#for i in [28,32,47,82,109,128,180,198,229,235,240,241,264,274,300,303,306,330,358,359,368,
#409,485,502,564,573,593,596,612,653,680,697,727,792,863,888,927,1002,1018,1051,1079,1091,1107,1118,
#1159,1164,1173,1195,1299,1301]:
    
 #   all_data.iloc[i,0] = 22.0

# print(all_data.Age[all_data["Name_ 5"]==1].isnull())



##name is man
#print("\n")
#print("manの平均　: "+str(all_data.Age[all_data["Name_ 1"]==1].mean()))
#print("manの中央値　: "+str(all_data.Age[all_data["Name_ 1"]==1].median()))
#print("manの最大値 : "+str(all_data.Age[all_data["Name_ 1"]==1].max()))
#print("manの最小値 : "+str(all_data.Age[all_data["Name_ 1"]==1].min()))

#all_data.Age.fillna(32.4,inplace=True)
      
#print(all_data.Age.isnull().sum())
print("\n")
print("*"*30)
print("Ticket data")
print("*"*30)
all_data.drop("Ticket",axis=1,inplace=True)
print("Ticket data droped. Because the thing learned for Ticket data is no. cheap or noble? this learn for Fare data.")
print("\n")
print("*"*30)
print("Fare data")
print("*"*30)
print("Pclass of Fare's NaN : "+str(all_data.Pclass[all_data["PassengerId"]==1044]))
print("Fare of Pclass'3 : "+str(all_data.Fare[all_data["Pclass"]==3].mean()))
print("Fare of Pclass'3 : "+str(all_data.Fare[all_data["Pclass"]==3].median()))
print("Fare of Pclass'3 : "+str(all_data.Fare[all_data["Pclass"]==3].max()))
print("Fare of Pclass'3 : "+str(all_data.Fare[all_data["Pclass"]==3].min()))

#plt.scatter(all_data.Fare,all_data["Pclass"])
#plt.savefig("Fare scatter")

#print(all_data.Fare.max())
#print(all_data.PassengerId[all_data["Fare"]==512.3292])
print(all_data[all_data["PassengerId"]==259])

all_data.Fare.fillna(all_data.Fare.median(),inplace=True)
print("\n")
print("*"*30)
print("Embarked data")
print("*"*30)
print(all_data.Embarked.value_counts(dropna=False))
print("\n")
print(all_data[all_data["Embarked"].isnull()])
print("\n")
print("S のFareについて")
print(all_data.Fare[all_data["Embarked"]=="S"].mean())
print(all_data.Fare[all_data["Embarked"]=="S"].max())
print(all_data.Fare[all_data["Embarked"]=="S"].min())
print("\n")
print("C のFareについて")
print(all_data.Fare[all_data["Embarked"]=="C"].mean())
print(all_data.Fare[all_data["Embarked"]=="C"].max())
print(all_data.Fare[all_data["Embarked"]=="C"].min())
print("\n")
print("Q のFareについて")
print(all_data.Fare[all_data["Embarked"]=="Q"].mean())
print(all_data.Fare[all_data["Embarked"]=="Q"].max())
print(all_data.Fare[all_data["Embarked"]=="Q"].min())

all_data.Embarked.fillna(all_data.Embarked.mode()[0],inplace=True)
all_data = pd.get_dummies(data=all_data,columns=["Embarked"])

print("\n")
print("*"*30)
print("Cabin data")
print("*"*30)
#print(all_data.Cabin.value_counts())
##ABCDEFTG <- 種類
#C>107
#B>122.3
#E>54
#A.41.24
#D>53
#F>18
#T>35
#G14

#all_data.Cabin.fillna("N",inplace=True)
#all_data.Cabin = [i[0] for i in all_data.Cabin]

#print(all_data.Fare[all_data["Cabin"]=="T"].mean())

k=0

for i in all_data.Cabin:
    if i == "N":
        if all_data.iloc[k,2] >= 122:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","B")
        elif all_data.iloc[k,2] >= 105:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","C")
        elif all_data.iloc[k,2] >= 54:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","E")
        elif all_data.iloc[k,2] >= 53:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","D")
        elif all_data.iloc[k,2] >= 41:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","A")
        elif all_data.iloc[k,2] >= 33:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","T")
        elif all_data.iloc[k,2] >= 15:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","F")
        else:
            all_data.iloc[k,1] = all_data.iloc[k,1].replace("N","G")
    k += 1

#all_data = pd.get_dummies(data=all_data,columns=["Cabin"])
all_data.drop("Cabin",axis=1,inplace=True)

all_data["Family_size"] = all_data["SibSp"] + all_data["Parch"]+1
all_data["IsAlone"] = 1
all_data['IsAlone'].loc[all_data['Family_size'] > 1] = 0

print("\n")
print("Age data")
print("*"*30)


all_data = all_data.drop("PassengerId",axis=1)

df_train = all_data[:891]
df_test = all_data[891:]

from sklearn.ensemble import RandomForestRegressor

def age_not(df):

    Age_N = df[df.Age.isnull()]
    Age_N_2 = Age_N.drop("Age",axis=1)
    Age = df[df.Age.notnull()]

    X = Age.drop("Age",axis=1)
    y = Age.Age

    RFR = RandomForestRegressor(random_state=0)
    RFR.fit(X,y)
    answer = RFR.predict(Age_N_2)
    df.loc[df.Age.isnull(), "Age"] = answer

    return df

age_not(df_train)
age_not(df_test)


print("\n")
print("*"*30)
print("study start")
print("\n")

PassengerId = test.PassengerId

X = df_train
y = train.Survived

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
df_test = sc.transform(df_test)

from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(C=1.0,kernel="linear")
clf.fit(X_train,y_train)
R_0 = clf.predict(X_test)
print("SVM : "+str(accuracy_score(R_0,y_test)))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', C=100, random_state=0)
lr.fit(X_train,y_train)
R_1 = lr.predict(X_test)
print("Logistic回帰 : "+str(accuracy_score(R_1,y_test)))

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train,y_train)
R_2 = random_forest.predict(X_test)
print("RandomForest : "+str(accuracy_score(R_2,y_test)))

from sklearn.neighbors import KNeighborsClassifier

Max = 0

for i in range(1,len(X_train)):
    knc = KNeighborsClassifier(n_neighbors=15)
    knc.fit(X_train,y_train)
    R_3 = knc.predict(X_test)
    M = accuracy_score(R_3,y_test)
    if M > Max:
        k = i
        Max = M
print("K近傍法 : "+str(Max))
print(k)

###i == 105

print("\n")
print("*"*30)
print("submit")
print("*"*30)

test_predict = lr.predict(df_test)
data = {'PassengerId':PassengerId,'Survived':test_predict}
submission = pd.DataFrame(data=data,dtype=int)
submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)
submission.to_csv("titanic5_submission.csv",index = False)

###Ageの処理
###Cabinの処理




