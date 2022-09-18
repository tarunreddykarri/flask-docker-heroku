import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

# Reading the data
data = pd.read_csv('hiring.csv')

# Fill empty experience with 0
data.experience = data.experience.fillna('zero')

# Converting words to integer values


def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict[word]


data['experience'] = data['experience'].apply(lambda x: convert_to_int(x))



# Fill the missing test score with mean value
test_score = data['test_score(out of 10)']
test_score = test_score.fillna(
    test_score.mean())

# Split the data into features and labels this is a regression problem
y = data['salary($)']
X = data.drop(['salary($)'], axis=1)

# Train our model
lin_reg = LinearRegression()

# Fitting the training data
lin_reg.fit(X, y)

# Get the performance of our model
print(lin_reg.score(X, y))

# Save our model 
pickle.dump(lin_reg, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))