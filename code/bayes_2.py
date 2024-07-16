import numpy as np
import pandas as pd
from scipy.stats import norm

# File paths
data_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_train.csv'
label_train = '/Users/yuanweiyun/Desktop/EE6227/dataset/label_train.csv'
data_test = '/Users/yuanweiyun/Desktop/EE6227/dataset/data_test.csv'

# Dataloader
x_train = pd.read_csv(data_train, header=None)  # Assuming no header in the CSV
y_train = pd.read_csv(label_train, header=None)
y_train = y_train[0]
x_test = pd.read_csv(data_test)

# Prior probability
prior_l1 = len(x_train[y_train == 1]) / len(x_train)
prior_l2 = len(x_train[y_train == -1]) / len(x_train)


# Mean and variance
means = x_train.groupby(y_train).mean()
variances = x_train.groupby(y_train).var()
print(means[-1], means[1])
print(variances[-1],variances[1])

def gaussian_pdf(x, mean, variance):
    d = len(x)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(variance)), (x - mean))
    coeff = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(variance)))
    pdf_value = coeff * np.exp(exponent)
    return pdf_value

def classifier(sample):
    log_prob_label1 = gaussian_pdf(sample, means.loc[1], variances.loc[1]) + prior_l1
    log_prob_label2 = gaussian_pdf(sample, means.loc[-1], variances.loc[-1]) + prior_l2
    print(log_prob_label1, log_prob_label2)
    if log_prob_label1 > log_prob_label2:
        return 1
    else:
        return -1

predictions = x_test.apply(classifier, axis=1)
predictions.to_csv('/Users/yuanweiyun/Desktop/EE6227/predictions.csv', index=False)







