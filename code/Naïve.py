import numpy as np
import pandas as pd

# Load the data
data_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_train.csv'
data_test = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_test.csv'
label_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/label_train.csv'

x_train = pd.read_csv(data_train, header=None)
y_train = pd.read_csv(label_train, header=None)
x_test = pd.read_csv(data_test, header=None)

# priors
prior_l1 = np.mean(y_train == 1)
prior_l2 = np.mean(y_train == -1)
x_train_l1 = x_train[y_train[0] == 1]
x_train_l2 = x_train[y_train[0] == -1]

# mean and standard deviation
mean_l1 = x_train_l1.mean(axis=0)
std_dev_l1 = x_train_l1.std(axis=0)
mean_l2 = x_train_l2.mean(axis=0)
std_dev_l2 = x_train_l2.std(axis=0)

def gaussian_pdf(x, mean, std_dev):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * exponent

predictions = []
for index, sample in x_test.iterrows():
    likelihood_l1 = np.prod(gaussian_pdf(sample, mean_l1, std_dev_l1))
    likelihood_l2 = np.prod(gaussian_pdf(sample, mean_l2, std_dev_l2))
    posterior_l1 = likelihood_l1 * prior_l1
    posterior_l2 = likelihood_l2 * prior_l2
    if posterior_l1 > posterior_l2:
        predictions.append(1)
    else:
        predictions.append(-1)

predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
predictions_df.to_csv('/Users/yuanweiyun/Desktop/EE6227/predictions_1.csv', index=False,header= None)
