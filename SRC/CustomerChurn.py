# -*- coding: utf-8 -*-
"""
Created on  MAY 20 21:57:54 2025

@author: poulomi
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics
from sklearn import tree
from sklearn import feature_selection
from sklearn import neighbors
from sklearn import naive_bayes

df=pd.read_csv("d:/folder/churn_train.txt",sep=", ",engine='python')

df1=pd.read_csv("d:/folder/churn_test.txt",sep=", ",engine='python')
le=preprocessing.LabelEncoder()
le1=preprocessing.LabelEncoder()
df.info()
df["st"]=le.fit_transform(df['st'])
df["intplan"]=le.fit_transform(df['intplan'])
df["voice"]=le.fit_transform(df['voice'])
df["label"]=le.fit_transform(df['label'])
df.dropna(inplace=True)
Xtrain=df.drop("phnum",axis=1)
Xtrain=Xtrain.drop("label",axis=1)

ytrain=df[['label']]
Xtrain.describe()
ytrain.describe()
ytrain['label'].value_counts()



#feature extraction

df['st'].value_counts()
sns.boxplot(y="st",data=df,x="label")
df['st'].isnull().sum()
sns.countplot(y='st',data=df,hue='label')
sns.distplot(df['st'])

df['acclen'].value_counts()
sns.boxplot(y="acclen",data=df,x="label")
sns.distplot(df['acclen'])
qr=np.percentile(df['acclen'],[0,25,50,75,100])
qr


Xtrain['intplan'].value_counts()
sns.boxplot(y="intplan",data=df,x="label")
sns.distplot(df['intplan'])
sns.countplot(y='intplan',data=df,hue='label')

Xtrain['voice'].value_counts()
sns.countplot(y='voice',data=df,hue='label')
sns.distplot(df['voice'])
sns.boxplot(y="voice",data=df,x="label")

Xtrain['nummailmes'].value_counts()
sns.boxplot(y="nummailmes",data=df,x="label")
sns.distplot(df['nummailmes'])

Xtrain['tdmin'].value_counts()
sns.boxplot(y="tdmin",data=df,x="label")
sns.distplot(df['tdmin'])

Xtrain['tdcal'].value_counts()
sns.boxplot(y="tdcal",data=df,x="label")
sns.distplot(df['tdcal'])

Xtrain['tdchar'].value_counts()
sns.boxplot(y="tdchar",data=df,x="label")
sns.distplot(df['tdchar'])

Xtrain['temin'].value_counts()
sns.boxplot(y="temin",data=df,x="label")
sns.distplot(df['temin'])

Xtrain['tecal'].value_counts()
sns.boxplot(y="tecal",data=df,x="label")
sns.distplot(df['tecal'])

Xtrain['tecahr'].value_counts()
sns.boxplot(y="tecahr",data=df,x="label")
sns.distplot(df['tecahr'])
sns.countplot(x='intplan',data=df,hue='label')

Xtrain['tnmin'].value_counts()
sns.boxplot(y="tnmin",data=df,x="label")
sns.distplot(df['tnmin'])

Xtrain['tn cal'].value_counts()
sns.boxplot(y="tn cal",data=df,x="label")
sns.distplot(df['tn cal'])

Xtrain['tnchar'].value_counts()
sns.boxplot(y="tnchar",data=df,x="label")
sns.distplot(df['tnchar'])

Xtrain['timin'].value_counts()
sns.boxplot(y="timin",data=df,x="label")
sns.distplot(df['timin'])

Xtrain['intplan'].value_counts()
sns.boxplot(y="intplan",data=df,x="label")
sns.distplot(df['intplan'])
sns.countplot(y='intplan',data=df,hue='label')

Xtrain['tical'].value_counts()
sns.boxplot(y="tical",data=df,x="label")
sns.distplot(df['tical'])
sns.countplot(x='tical',data=df,hue='label')

Xtrain['tichar'].value_counts()
sns.boxplot(y="tichar",data=df,x="label")
sns.distplot(df['tichar'])

Xtrain['intplan'].value_counts()
sns.boxplot(y="intplan",data=df,x="label")
sns.distplot(df['intplan'])
sns.countplot(x='intplan',data=df,hue='label')

df['ncsc'].value_counts()
sns.boxplot(y="ncsc",data=df,x="label")
sns.distplot(df['ncsc'])
sns.countplot(x='ncsc',data=df,hue='label')

Xtrain=Xtrain.drop("st",axis=1)
Xtrain=Xtrain.drop("acclen",axis=1)
Xtrain=Xtrain.drop("tdcal",axis=1)

Xtrain=Xtrain.drop("tecal",axis=1)

Xtrain=Xtrain.drop("temin",axis=1)

Xtrain=Xtrain.drop("tnchar",axis=1)
Xtrain=Xtrain.drop("tdchar",axis=1)




ml=tree.DecisionTreeClassifier()
ml.fit(Xtrain,ytrain)
print("AUC:",metrics.roc_auc_score(ytrain,ml.predict(Xtrain)))
print("recall:",metrics.recall_score(ytrain,ml.predict(Xtrain)))

df1["st"]=le1.fit_transform(df1['st'])
df1["intplan"]=le1.fit_transform(df1['intplan'])
df1["voice"]=le1.fit_transform(df1['voice'])
df1["label"]=le1.fit_transform(df1['label'])
df1.dropna(inplace=True)
Xtest=df1.drop("phnum",axis=1)
Xtest=Xtest.drop("label",axis=1)
Xtest=Xtest.drop("st",axis=1)
Xtest=Xtest.drop("acclen",axis=1)
Xtest=Xtest.drop("tdcal",axis=1)

Xtest=Xtest.drop("tecal",axis=1)

Xtest=Xtest.drop("temin",axis=1)


Xtest=Xtest.drop("tnchar",axis=1)
Xtest=Xtest.drop("tdchar",axis=1)





ytest=df1[['label']]
ml=tree.DecisionTreeClassifier()
ml.fit(Xtrain,ytrain)
print("AUC:",metrics.roc_auc_score(ytest,ml.predict(Xtest)))
print("recall:",metrics.recall_score(ytest,ml.predict(Xtest)))
print("F1:",metrics.precision_score(ytest,ml.predict(Xtest)))
print("F1:",metrics.accuracy_score(ytest,ml.predict(Xtest)))
confmat=metrics.confusion_matrix(ytest,ml.predict(Xtest))
print(confmat)

#continouse heatmap

df2=df
df2=df2.drop('st',axis=1)
df2=df2.drop('intplan',axis=1)
df2=df2.drop('tichar',axis=1)
df2=df2.drop('voice',axis=1)
df2=df2.drop('label',axis=1)
sns.heatmap(df2.corr())




def modelstats(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames)
        
print(modelstats(Xtrain,Xtest,ytrain,ytest))
