#Import needed packages
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

#read data set
s = pd.read_csv('social_media_usage.csv')

#create clean function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

#create ss df
ss = s[['web1h','income', 'educ2', 'par', 'marital', 'gender', 'age']]

#create sm_li and drop web1h
ss['sm_li'] = clean_sm(ss['web1h'])
ss = ss.drop('web1h', axis=1)

#Replace string missing with NaN value to ensure NAs are dropped correctly later
ss['income'].where(ss['income'] < 10, np.nan, inplace=True)
ss['educ2'].where(ss['educ2'] < 9, np.nan, inplace=True)
ss['par'].where(ss['par'] < 3, np.nan, inplace=True)
ss['marital'].where(ss['marital'] < 3, np.nan, inplace=True)
ss['gender'].where(ss['gender'] < 3, np.nan, inplace=True)
ss['age'].where(ss['age'] < 99, np.nan, inplace=True)

#Drop NA values
ss = ss.dropna()

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age']]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# Initialize algorithm 
lr = LogisticRegression(class_weight = 'balanced')

# Fit algorithm to training data
lr.fit(X_train, y_train)

"## Predicting whether a person is a LinkedIn user."
"### Input your data for the person."

income = st.selectbox(label="What is the income level?",
options=('<10,000','10,000 < 20,000','20,000 < 30,000','30,000 < 40,000','40,000 < 50,000','50,000 < 75,000','75,000 < 100,000','100,000 < 150,000', '>150,000'))
if income == '<10,000':
    income1 = 1
elif income == '10,000 < 20,000':
    income1 = 2
elif income == '20,000 < 30,000':
    income1 = 3
elif income == '30,000 < 40,000':
    income1 = 4
elif income == '40,000 < 50,000':
    income1 = 5
elif income == '50,000 < 75,000':
    income1 = 6
elif income == '75,000 < 100,000':
    income1 = 7
elif income == '100,000 < 150,000':
    income1 = 8
elif income == '>150,000':
    income1 = 9


education = st.selectbox(label="What is the highest level of education completed?",
options=('Less than high school (Grades 1-8 or no formal schooling)','High school incomplete (Grades 9-11 or Grade 12 with NO diploma)', 'High school graduate (Grade 12 with diploma or GED certificate)','Some college, no degree (includes some community college)', 'Two-year associate degree from a college or university','Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)','Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)','Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'))
if education == 'Less than high school (Grades 1-8 or no formal schooling)':
    education1 = 1
elif education == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)':
    education1 = 2
elif education == 'High school graduate (Grade 12 with diploma or GED certificate)':
    education1 = 3
elif education == 'Some college, no degree (includes some community college)':
    education1 = 4
elif education == 'Two-year associate degree from a college or university':
    education1 = 5
elif education == 'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)':
    education1 = 6
elif education == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
    education1 = 7
elif education == 'Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)':
    education1 = 8

parent = st.selectbox(label="Are they a parent?",
options = ('Yes', "No"))
if parent == 'Yes':
    parent1 = 1
elif parent == 'No':
    parent1 = 2

married = st.selectbox(label="What is the marital status?",
options=('Married','Living with a partner','Divorced','Separated','Widowed','Never been married'))
if married == 'Married':
    married1 = 1
elif married == 'Living with a partner':
    married1 = 2
elif married == 'Divorced':
    married1 = 3
elif married == 'Separated':
    married1 = 4
elif married == 'Widowed':
    married1 = 5
elif married == 'Never been married':
    married1 = 6

gender = st.selectbox(label="What is their gender?",
options=('Male','Female','Other'))
if gender == 'Male':
    gender1 = 1
elif gender == 'Female':
    gender1 = 2
elif gender == 'Other':
    gender1 = 3


age = st.slider("Please select the age")


# New data for features: Income, Education, Parent, Marital, Gender, and Age
test_person = [income1, education1, parent1, married1, gender1, age]

# Predict class, given input features
predicted_class = lr.predict([test_person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([test_person])
probs = round(probs,2)*100

st.write(f"For a person with income of {income}, education level of {education}, parent status of {parent}, marital status of {married}, gender {gender} and age {age}.")

# Print predicted class and probability
st.write(f"The predicted class is: {predicted_class[0]}. (0 is not a user, 1 is a user)") # 0 = not a linkedin user, 1 = linkedin user
st.write(f"The probability that this person is a LinkedIn user: {probs[0][1]}%")