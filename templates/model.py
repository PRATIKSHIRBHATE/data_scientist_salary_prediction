# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('data_scientist_salary.csv')

columns = ['experience', 'degree_percentage', 'post_graduation', 'iit_nit',
       'designation', 'aws_gcp_azure_ml_ops', 'tableau_powerbi', 'salary']
       
for col in ['post_graduation', 'iit_nit', 'designation', 'aws_gcp_azure_ml_ops', 'tableau_powerbi']:
    dataset[col] = dataset[col].str.lower()

for col in ['post_graduation', 'iit_nit', 'aws_gcp_azure_ml_ops', 'tableau_powerbi']:
    dataset[col] = dataset[col].replace({"no":0, "yes":1})

dataset["designation"] = dataset["designation"].replace({"data_scientist":0, "sr_data_scientist":1})

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X.values, y.values)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4.6, 82, 1,1,1,0,1]]))
