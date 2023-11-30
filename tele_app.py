#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import category_encoders as ce

# Load the pre-trained models and scaler
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Load the dataset
train = pd.read_csv('/home/futures/Downloads/train.csv')

train


# In[ ]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from xgboost import XGBClassifier
import streamlit as st
from PIL import Image
import pickle

# Load your model and data preprocessing objects
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

hash_state = ce.HashingEncoder(cols='state')

# Function to preprocess input data
def preprocess_input(input_dict):
    # Add your preprocessing logic here
    # Ensure that the preprocessing steps match those used during model training
    return dv.transform([input_dict])

# Function to predict using the loaded model
def predict_churn(X):
    return model.predict_proba(X)[0, 1]

# Function to remove outliers
def remove_outliers(train, labels):
    for label in labels:
        q1 = train[label].quantile(0.25)
        q3 = train[label].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        train[label] = train[label].mask(train[label] < lower_bound, train[label].median(), axis=0)
        train[label] = train[label].mask(train[label] > upper_bound, train[label].median(), axis=0)

    return train

# Load the dataset
train = pd.read_csv('/home/futures/Downloads/train.csv')
test = pd.read_csv('/home/futures/Downloads/test.csv')

# ... (rest of the code remains the same up to the Streamlit section)







# In[ ]:



# Streamlit code
def main():
    image = Image.open('/home/futures/Downloads/icone.png')
    image2 = Image.open('/home/futures/Downloads/image.png')
    st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    
    
    if add_selectbox == 'Online':
        state = st.selectbox('State:', train['state'].unique())
        account_length = st.number_input('Account Length:', min_value=0)
        area_code = st.selectbox('Area Code:', train['area_code'].unique())
        international_plan = st.selectbox('International Plan:', ['no', 'yes'])
        voice_mail_plan = st.selectbox('Voice Mail Plan:', ['no', 'yes'])
        number_vmail_messages = st.number_input('Number of Voice Mail Messages:', min_value=0)
        total_day_minutes = st.number_input('Total Day Minutes:', min_value=0)
        total_day_calls = st.number_input('Total Day Calls:', min_value=0)
        total_day_charge = st.number_input('Total Day Charge:', min_value=0)
        total_eve_minutes = st.number_input('Total Evening Minutes:', min_value=0)
        total_eve_calls = st.number_input('Total Evening Calls:', min_value=0)
        total_eve_charge = st.number_input('Total Evening Charge:', min_value=0)
        total_night_minutes = st.number_input('Total Night Minutes:', min_value=0)
        total_night_calls = st.number_input('Total Night Calls:', min_value=0)
        total_night_charge = st.number_input('Total Night Charge:', min_value=0)
        total_intl_minutes = st.number_input('Total International Minutes:', min_value=0)
        total_intl_calls = st.number_input('Total International Calls:', min_value=0)
        total_intl_charge = st.number_input('Total International Charge:', min_value=0)
        number_customer_service_calls = st.number_input('Number of Customer Service Calls:', min_value=0)
    
    # Rest of the code remains the same

        input_dict = {
            "state": state,
            "account_length": account_length,
            "area_code": area_code,
            "international_plan": international_plan,
            "voice_mail_plan": voice_mail_plan,
            "number_vmail_messages": number_vmail_messages,
            "total_day_minutes": total_day_minutes,
            "total_day_calls": total_day_calls,
            "total_day_charge": total_day_charge,
            "total_eve_minutes": total_eve_minutes,
            "total_eve_calls": total_eve_calls,
            "total_eve_charge": total_eve_charge,
            "total_night_minutes": total_night_minutes,
            "total_night_calls": total_night_calls,
            "total_night_charge": total_night_charge,
            "total_intl_minutes": total_intl_minutes,
            "total_intl_calls": total_intl_calls,
            "total_intl_charge": total_intl_charge,
            "number_customer_service_calls": number_customer_service_calls,
         }

    
    
    
    
        if st.button("Predict"):
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
            st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = dv.transform(data)
            y_pred = model.predict_proba(X)[:, 1]
            churn = y_pred >= 0.5
            churn = churn.astype(bool)
            result_df = pd.DataFrame({'Churn Prediction': churn, 'Risk Score': y_pred})
            st.write(result_df)

            
            
            
            
            
if __name__ == '__main__':
    main()


# Rest of the Streamlit code remains the same up to the 'if __name__ == '__main__':' line


# In[ ]:




