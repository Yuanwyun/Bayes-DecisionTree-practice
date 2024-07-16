import numpy as np
import pandas as pd

# Load the data
data_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_train.csv'
data_test = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_test.csv'
label_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/label_train.csv'

x_train = pd.read_csv(data_train, header=None)
y_train = pd.read_csv(label_train, header=None)[0]
x_test = pd.read_csv(data_test, header=None)

x_train_l1 = x_train[y_train == 1]
x_train_l2 = x_train[y_train == -1]

mean_l1 = x_train_l1.mean(axis=0)
mean_l2 = x_train_l2.mean(axis=0)

Sw_l1 = np.cov(x_train_l1, rowvar=False)
Sw_l2 = np.cov(x_train_l2, rowvar=False)

Sw_inverse = np.linalg.inv(Sw_l1 + Sw_l2)
# weights for linear discriminant function
weights = np.dot(Sw_inverse, (mean_l1 - mean_l2))

predictions = np.sign(np.dot(x_test, weights))
predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
predictions_df.to_csv('/Users/yuanweiyun/Desktop/EE6227/predictions_2.csv', index=False, header= None)
