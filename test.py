import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

data = pd.read_csv('creditcard.csv')

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud],axis=0)

X = data.drop(columns="Class", axis = 1)
Y = data["Class"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2)

model = LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)

train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test),Y_test)


#web app used to visualize

st.title("credit card fraud detection model")
input_df = st.text_input('enter all required feature values')
input_df_splited = input_df.split(',')

submit = st.button("submit")

if submit:
    features = np.asarray(input_df_splited, dtype=np.float64)
    prediction = model.predict(features.reshape(1, -1))

    if prediction[0] == 0:
        st.write("✅ This transaction is likely **Genuine**")
    else:
        st.write("🚨  Alert! This transaction is likely **Fraudulent**")