#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score
import pandas as pd


# In[9]:




st.title('Streamlit Example')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",("Credit_card","Dummy"))

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('GB', 'SVM', 'Random Forest')
)

def get_dataset(name):
    data = None
    name == 'Credit card'
    data = pd.read_csv("default of credit card clients.csv")
    X = data.iloc[:, 1:-1]
    y = data.iloc[:,-1]         
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'GB':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'GB':
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

st.write('Enter the values for prediction')
LIMIT_BAL = st.number_input("LIMIT_BAL")
SEX = st.number_input("SEX")
EDUCATION=st.number_input("EDUCATION")
MARRIAGE=st.number_input("MARRIAGE")
AGE=st.number_input("AGE")
PAY_0=st.number_input("PAY_0")
PAY_2=st.number_input("PAY_2")
PAY_3=st.number_input("PAY_3")
PAY_4=st.number_input("PAY_4")
PAY_5=st.number_input("PAY_5")
PAY_6=st.number_input("PAY_6")
BILL_AMT1=st.number_input("BILL_AMT1")
BILL_AMT2=st.number_input("BILL_AMT2")
BILL_AMT3=st.number_input("BILL_AMT3")
BILL_AMT4=st.number_input("BILL_AMT4")
BILL_AMT5=st.number_input("BILL_AMT5")
BILL_AMT6=st.number_input("BILL_AMT6")
PAY_AMT1=st.number_input("PAY_AMT1")
PAY_AMT2=st.number_input("PAY_AMT2")
PAY_AMT3=st.number_input("PAY_AMT3")
PAY_AMT4=st.number_input("PAY_AMT4")
PAY_AMT5=st.number_input("PAY_AMT5")
PAY_AMT6=st.number_input("PAY_AMT6")
client_data = [LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]
data= np.array(list(client_data)).reshape(1,-1)

clf.predict(data)
if clf.predict(data)[0] == 1:
    st.write("the customer is default")
else:
    st.write("Not default")


#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)


