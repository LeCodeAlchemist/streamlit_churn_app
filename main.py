import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb


st.write("""
# Customer Churn Prediction App
This app predicts if a customer will leave the company or not!
""")

st.write('## Note')
st.write('If you are going to supply a .csv file make sure at has the columns shown in the example below')
st.write('The columns dont necessarily have to be in the same order. Just make sure you supply them')
example_data = {
    'customerID': ['7590-VHVEG'],
    'gender': ['Male'], 
    'SeniorCitizen': ['Yes'],
    'Partner': ['No'],
    'Dependents': ['No'],
    'MultipleLines': ['Yes'],
    'Contract': ['One_year'],
    'PaperlessBilling': ['No'],
    'PaymentMethod': ['Electronic_check'],
    'MonthlyCharges': [19.00],
    'TotalCharges': [150.65]
}

example_df = pd.DataFrame(example_data)
st.dataframe(example_df)

st.write('In case you want to filter your .csv file using the columns, you can copy them below.')
st.code("columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract',  'MultipleLines','PaperlessBilling', 'PaymentMethod','TotalCharges', 'MonthlyCharges', 'Churn']", language='python')

st.sidebar.header('User Input Features')

# Formats string inputs for test data matches training data
def format_value(value: str):
    if value == 'Month to month':
        return value.replace(' ', '-')
    else:
        return value.replace(' ', '_')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write('## Input DataFrame ')
    st.dataframe(input_df)

    customer_ids = input_df.loc[:, 'customerID']
    input_df.drop(columns=['customerID'], inplace=True)
else:
    def user_input_features():
        customer_id = st.sidebar.text_input('Customer ID', help="Enter the Customer ID")
        gender = st.sidebar.selectbox('Gender',('Male','Female'))
        senior_citizen = st.sidebar.selectbox('Senior Citizen',('Yes','No'))
        partner = st.sidebar.selectbox('Partner',('Yes','No'))
        dependents = st.sidebar.selectbox('Dependents',('Yes','No'))
        contract = st.sidebar.selectbox('Contract',('Month to month','One year', 'Two year'))
        multiple_lines = st.sidebar.selectbox('Multiple Lines',('No phone service','Yes', 'No'))
        paperless_billing = st.sidebar.selectbox('PaperlessBilling',('Yes','No'))
        payment_method = st.sidebar.selectbox('PaymentMethod',('Electronic check','Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
        
        monthly_charges = st.sidebar.slider('Monthly Charges ($)', 0.0, 200.0)
        total_charges = st.sidebar.slider('Total Charges ($)', 0.0, 10000.0)

        contract = format_value(contract)
        multiple_lines = format_value(multiple_lines)
        paperless_billing = format_value(paperless_billing)
        payment_method = format_value(payment_method)
        monthly_charges = round(monthly_charges, 2)
        total_charges = float("{:.2f}".format(total_charges))

        new_data = {
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MultipleLines': [multiple_lines],
            'TotalCharges': [total_charges],
            'MonthlyCharges': [monthly_charges]
        }
        input_df = pd.DataFrame(new_data)
        st.write('## Input DataFrame')
        st.dataframe(input_df)
        return input_df, customer_id

    input_df, customer_id = user_input_features()

# Combines user input features with a subset of the telco churn dataset
# This will be useful for the encoding of user input data
def preprocess_data(new_df, number_of_rows):
    selected_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract',  'MultipleLines','PaperlessBilling', 'PaymentMethod','TotalCharges', 'MonthlyCharges', ]
    # Read a subset of the origina training dataset to format the input for prediction
    preprocess_df = pd.read_csv('data/preprocess.csv', usecols=selected_features)

    # Join the dataframe with input data to the original training dataset to get all used features  
    combined = pd.concat([new_df, preprocess_df], ignore_index=True, axis=0)
    
    # Remove spaces  
    combined.replace(' ', '_', regex=True, inplace=True)


    categories = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MultipleLines']
    # Encode the data to get all features used in training  
    combined_encoded = pd.get_dummies(combined, columns=categories)
    
    input_df_encoded = combined_encoded.head(number_of_rows)
    return input_df_encoded 



if uploaded_file is not None:
    df_length = input_df.shape[0]
    input_df_encoded = preprocess_data(input_df, df_length)
    st.write('## Encoded Input Dataframe')
    st.dataframe(input_df_encoded)

else:
    input_df_encoded = preprocess_data(input_df, 1)
    st.write('## Encoded Input Dataframe')
    st.dataframe(input_df_encoded)



# Load and use model
model = xgb.XGBClassifier()
model.load_model('xgb_customer_churn.json')

features = model.get_booster().feature_names
input_df_encoded = input_df_encoded[features]

if uploaded_file is not None:
    prediction = pd.Series(model.predict(input_df_encoded))
    prediction = prediction.astype(str)
    prediction.replace(to_replace='0', value='No', inplace=True)
    prediction.replace(to_replace='1', value='Yes', inplace=True)

    probabilities = pd.DataFrame(model.predict_proba(input_df_encoded))
    probabilities.columns = ['Probability(Not Churning)', 'Probability(Churning)']


    # Join the results to the input dataframe to show the churn value for each user.
    input_df['Churn'] = prediction
    input_df['customerID'] = customer_ids

    columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract',  'MultipleLines','PaperlessBilling', 'PaymentMethod','TotalCharges', 'MonthlyCharges', 'Churn']

    input_df= input_df[columns]
    output = pd.concat([input_df, probabilities], axis=1)

    st.write('## Churn Prediction result')
    st.write('The dataframe below shows the churn prediction result as well as the relative probabilities of of churning or not churning')
    st.dataframe(output)

else:

    prediction = model.predict(input_df_encoded)
    probabilities = pd.DataFrame(model.predict_proba(input_df_encoded))

    # Rename the columns
    probabilities.columns = ['Probability of Not Churning', 'Probability of Churning']

    churn_value = prediction[0]

    st.write('## Churn Prediction Result')
    if churn_value == 0:
        st.write('The predicted churn value is ', churn_value)

        if customer_id.strip() == '':
            st.write('Therefore the customer will not leave the company.')
        else:    
            st.markdown(f'Therefore the customer with ** CustomerID={customer_id} ** will not leave the company')
    else:
        st.write('The predicted churn value is ', churn_value)
        if customer_id.strip() == '':
            st.write('The customer will likely leave the company')
        else:
            st.markdown(f'Therefore the customer with ** CustomerID={customer_id} ** will likely leave the company')
    
    st.write('### The probabilities of leaving or not leaving are shown ** below **')
    st.dataframe(probabilities)
