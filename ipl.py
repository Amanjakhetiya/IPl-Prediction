import pandas as pd
import numpy as np

df = pd.read_csv('matches.csv')

new_df = df[['team1','team2','toss_decision','toss_winner','winner']]

new_df.dropna(inplace=True)

all_teams = {}
cnt = 0

for i in range(len(df)):
    if df.loc[i]['team1'] not in all_teams:
        all_teams[df.loc[i]['team1']] = cnt
        cnt+=1

    if df.loc[i]['team2'] not in all_teams:
        all_teams[df.loc[i]['team2']] = cnt
        cnt+=1

X = new_df[['team1','team2','toss_decision','toss_winner']]
y = new_df[['winner']]

encoded_teams = {w:k for k,w in all_teams.items()}

X = np.array(X)
y = np.array(y)

for i in range(len(X)):
    X[i][0] = all_teams[X[i][0]]
    X[i][1] = all_teams[X[i][1]]
    X[i][3] = all_teams[X[i][3]]

    y[i][0] = all_teams[y[i][0]]

fb = {'field' : 0,'bat' : 1}

for i in range(len(X)):
    X[i][2] = fb[X[i][2]]



for i in range(len(X)):
    if X[i][3] == X[i][0]:
        X[i][3] = 0
    else:
        X[i][3] = 1


ones = 0

for i in range(len(y)):
    if y[i][0] == X[i][1]:
        if ones < 370:
            ones+=1
            y[i][0] = 1
        else:
            t = X[i][0]
            X[i][0] = X[i][1]
            X[i][1] = t
            y[i][0] = 0
    else:
        y[i][0] = 0

X = np.array(X,dtype = 'int64')
y = np.array(y,dtype = 'int64')

y = y.ravel()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

from sklearn.svm import SVC
model1 = SVC().fit(X_train,y_train)
model1.score(X_test,y_test)

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier().fit(X_train,y_train)
model2.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
model3.score(X_test,y_test)

import pickle as pkl

with open('model.pkl','wb') as f:
    pkl.dump(model3,f)

with open('vocab.pkl','wb') as f:
    pkl.dump(encoded_teams,f)

with open('inv_vocab.pkl','wb') as f:
    pkl.dump(all_teams,f)



