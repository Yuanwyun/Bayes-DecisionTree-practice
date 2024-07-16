import numpy as np
import pandas as pd

data_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_train.csv'
data_test = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_test.csv'
label_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/label_train.csv'
#dataloader
x_train = pd.read_csv(data_train,header = None).values
y_train = pd.read_csv(label_train,header= None)
y_train = y_train[0]
x_test = pd.read_csv(data_test, header = None)

# prior probability 
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

prior_l1 = len(x_train[y_train == 1]) / len(x_train)
prior_l2 = len(x_train[y_train == -1]) / len(x_train)
# mean and variance
#means = x_train.groupby(y_train).mean()
mean_l1 = x_train[y_train == 1].mean(axis=0)
mean_l2 = x_train[y_train == -1].mean(axis=0)

cov_l1 = np.cov(x_train[y_train == 1], rowvar=False) 
cov_l2 = np.cov(x_train[y_train == -1], rowvar=False)
print(cov_l1,cov_l2)

# cause the mean matrix is 1*5, and the variance matrix is 5*5
def gaussian_pdf(x, mean, variance):
    d = len(x)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(variance)), (x - mean))
    coeff = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(variance)))
    pdf_value = coeff * np.exp(exponent)
    return pdf_value

def classifier(sample):
    label1 =  gaussian_pdf(sample, mean_l1, cov_l1 ) * prior_l1
    label2 = gaussian_pdf(sample, mean_l2, cov_l2) * prior_l2
    print(label1,label2)
    if label1 > label2:
        return 1
    else:
        return -1

predictions = x_test.apply(classifier, axis=1)
predictions.to_csv('/Users/yuanweiyun/Desktop/EE6227/predictions.csv', index=False, header=None)