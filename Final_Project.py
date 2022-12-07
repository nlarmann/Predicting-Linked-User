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

"## Predicting whether a person is a linkedin user"
"### Input your data for the person. Please  reference the legend below"

income = st.selectbox(label="What is the income level?",
options=(1, 2, 3, 4, 5, 6, 7, 8,9))

education = st.selectbox(label="What is the highest level of education completed?",
options=(1, 2, 3, 4, 5, 6, 7, 8))

parent = st.selectbox(label="Are they a parent?",
options=(1, 2))

married = st.selectbox(label="Are they married?",
options=(1, 2, 3, 4, 5, 6))

gender = st.selectbox(label="What is their gender?",
options=(1, 2, 3))

age = st.slider("x")


# New data for features: Income, Education, Parent, Marital, Gender, and Age
test_person = [income, education, parent, married, gender, age]

# Predict class, given input features
predicted_class = lr.predict([test_person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([test_person])

st.write(f"For a person with income of {income}, education of {education}, parent status of {parent}, marital status of {married}, gender {gender} and age {age}.")

# Print predicted class and probability
st.write(f"The predicted class is: {predicted_class[0]}. (0 is not a user, 1 is a user)") # 0 = not a linkedin user, 1 = linkedin user
st.write(f"The probability that this person is a LinkedIn user: {probs[0][1]}")

#Create legend
'#### Income:'
'1: < 10,000'
'2: 10,000 to < 20,000'
'3: 20,000 to < 30,000,'
'4: 30,000 to < 40,000'
'5: 40,000 to < 50,000'
'6: 50,000 to < 75,000'
'7: 75,000 to < 100,000'
'8: 100,000 to < 150,000'
'9: above 150,000'

'#### Highest level of education'
'1: Less than high school (Grades 1-8 or no formal schooling)'
'2: High school incomplete (Grades 9-11 or Grade 12 with NO diploma)'
'3: High school graduate (Grade 12 with diploma or GED certificate)'
'4: Some college, no degree (includes some community college)'
'5: Two-year associate degree from a college or university'
'6: Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)'
'7: Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)'
'8: Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'

'#### Is the person a parent?'
'1:	Yes'
'2: No'

'#### What is the person marital status?'
'1: Married'
'2:	Living with a partner'
'3:	Divorced'
'4:	Separated'
'5:	Widowed'
'6:	Never been married'

'#### What is their gender?'
'1: Male'
'2: Female'
'3: Other'