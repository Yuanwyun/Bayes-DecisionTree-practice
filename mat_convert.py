import scipy.io
import pandas as pd

# Load the .mat file
data = scipy.io.loadmat('/Users/yuanweiyun/Desktop/EE6227/label_train.mat')

# Assume 'data' is a dictionary that contains ndarrays or similar structures
# Convert these to pandas DataFrame (you will need to adjust this code depending on the structure of your .mat file)
df = pd.DataFrame(data['label_train'])  # Replace 'your_variable_name' with the relevant variable name


#df.to_excel('data_test.xlsx', index=False, engine='openpyxl')

df.to_csv('label_train.csv', index=False)
