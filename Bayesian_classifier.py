import pickle
import numpy as np
import scipy.stats
from scipy.stats import norm
from skimage import transform as skt
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

#classification accuracy
def class_acc(pred, gt):
    N = len(gt)
    corr = 0
    for i in range(len(gt)):
        if (pred[i]) == (gt[i]):
            corr = corr + 1
    tot_correct = corr/(N)
    print(f'\n Classication accuracy: {tot_correct * 100:.2f}%')


#resizing color array (can be used for both TEST and TRAINING)
def cifar10_color(X):
    #:param X: original images X(50000 x 3072)
    #:return: resized images X(50000 x 3)
    Xp = []
    for index in range(len(X)):
        re_pic = []
        Red = X[index][0:1024]
        Green = X[index][1024:2049]
        Blue = X[index][2049:]
        re_pic.append(np.mean(Red))
        re_pic.append(np.mean(Green))
        re_pic.append(np.mean(Blue))
        Xp.append(re_pic)
    return Xp

#reszing color array (can be used for both TEST and TRAINING)
def cifar10_NxN_color(X, size = 1):
    if size > 1:
        Xr = np.array([skt.resize(image, output_shape=(size, size), preserve_range=True) for image in X]).reshape(X.shape[0], size * size*3)
        return Xr
    Xr = np.array([skt.resize(data, output_shape=(size,size), preserve_range=True) for data in X]).reshape(X.shape[0], 3)
    return Xr

#computing the normal distribution parameters of all training data (NAIVE)
def cifar_10_naivebayes_learn(Xp, Y):
    muRGB_10 = []
    sigmaRGB_10 = []
    p_10 = []
    for n in range(10):
        index_list = []
        trainData_one_class = []
        for i in range(len(Y)):
            if Y[i] == n:
                index_list.append(i)
                trainData_one_class.append(Xp[i])
        muRGB = np.mean(trainData_one_class, axis= 0)
        sigmaRGB = np.var(trainData_one_class, axis = 0)**(1/2)
        muRGB_10.append(muRGB)
        sigmaRGB_10.append(sigmaRGB)
        p_10.append(0.10)
    #print(muRGB_10, sigmaRGB_10)
    return muRGB_10, sigmaRGB_10, p_10

#normal distribution function (no call directly)
def normal_dist(x, mu, sigma):
    '''
    :param x: 1x3
    :param mu: 1x3
    :param sigma: 1x3
    :return:
    '''
    prob = 1
    for i in range(3):
        temp = norm.pdf(x, mu[i], sigma[i])
        prob *= temp
    return prob

#getting the Bayesian optimal class for the sample x
def cifar10_classifier_naivebayes(x, mu, sigma, p):
    '''
    x: 1x3
    mu: 10x3
    sigma: 10x3
    p: 10x1
    '''
    best = 0
    best_class = 0
    for index in range(10):
        prob = normal_dist(x, mu[index], sigma[index]) * p[index]
        if (prob > best).all():
            best = prob
            best_class = index
    return best_class

##computing the normal distribution parameters of all training data (BETTER)
def cifar10_bayes_learn(Xf, Y, size = 1):
    muRGB_10 = []
    sigmaRGB_10 = []
    p_10 = []
    for n in range(10):
        index_list = []
        trainData_one_class = []
        for i in range(len(Y)):
            if Y[i] == n:
                index_list.append(i)
                trainData_one_class.append(Xf[i])
        trainData_one_class = np.asarray(trainData_one_class)
        muRGB = np.mean(trainData_one_class, axis= 0)
        sigmaRGB = np.cov(trainData_one_class, rowvar= False)
        muRGB_10.append(muRGB)
        sigmaRGB_10.append(sigmaRGB)
        p_10.append(0.10)
    sigmaRGB_10 = np.asarray(sigmaRGB_10)
    if size > 1:
        return muRGB_10, sigmaRGB_10.reshape(10, size*size*3, size*size*3), p_10
    return muRGB_10, sigmaRGB_10.reshape(10, 3, 3), p_10

#getting the Bayesian optimal class (BETTER) for the sample x
def cifar10_classifier_bayes(x, mu, sigma, p):
    '''
    x: 1x3
    mu: 10x3
    sigma: 10x3x3
    p: 10x1
    '''
    best = 0
    best_class = 0
    for index in range(10):
        prob = scipy.stats.multivariate_normal.pdf(x, mu[index], sigma[index]) * p[index]
        if (prob > best).all():
            best = prob
            best_class = index
    return best_class

labeldict = unpickle('D:\\UNI ASSIGNMENTS\\intro to pattern recog n ML\\EXERCISE 3\\cifar-10-python\\cifar-10-batches-py\\batches.meta')
label_names = labeldict["label_names"]

#storing testing data and its corresponding labels
datadict = unpickle('D:\\UNI ASSIGNMENTS\\intro to pattern recog n ML\\EXERCISE 3\\cifar-10-python\\cifar-10-batches-py\\test_batch') #testing data
X = datadict["data"] #unclassified data
Y = datadict["labels"] #ground truth label of unclassified data
X = X.astype("float32")
#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
print(X.shape)
print(len(Y))

#storing traning data and its corresponding labels
tr_data = np.array([]) #training data
tr_label = np.array([]) #training data's labels
for i in range(1, 6):
    training_data = unpickle(f'D:\\UNI ASSIGNMENTS\\intro to pattern recog n ML\\EXERCISE 3\\cifar-10-python\\cifar-10-batches-py\\data_batch_{i}')
    tr_data = np.append(tr_data, training_data["data"])
    tr_label = np.append(tr_label, training_data["labels"])
tr_data = tr_data.reshape(50000, 3072).astype("float32")
#tr_data = tr_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
print(tr_data.shape)
tr_label = np.array(tr_label)

#resizing training data
tr_data_reszied = cifar10_color(tr_data)
#resizing test data
X_resize = cifar10_color(X)

#NAIVE
#computing normal distribution values of training data from all 10 classes
muRGB_10, sigmaRGB_10, p_10 = cifar_10_naivebayes_learn(tr_data_reszied, tr_label)
#classifying the images with Naive Bayes classifier
tested_class = []
for image in X_resize:
    best_class = cifar10_classifier_naivebayes(image, muRGB_10, sigmaRGB_10, p_10)
    tested_class.append(best_class)
#classification accuracy for Naive Bayes classifier
print("Classification accuracy for Naive Bayes classifier")
class_acc(tested_class, Y)

#BETTER BAYES
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
tr_data = tr_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
#resizing training data
tr_data_reszied = cifar10_NxN_color(tr_data, 1)
#resizing test data
X_resize = cifar10_NxN_color(X, 1)
#computing multivariate normal dist values of training data from all 10 classes
muRGB_10_better, sigmaRGB_10_better, p_10 = cifar10_bayes_learn(tr_data_reszied, tr_label)
#classifying the images with Bayes classifier
tested_class = []
count = 1
for image in X_resize:
    best_class = cifar10_classifier_bayes(image, muRGB_10_better, sigmaRGB_10_better, p_10)
    tested_class.append(best_class)
    #print(count)
    count += 1
#classification accuracy for better Bayes classifier
print("Classification accuracy for 'better' Bayes classifier")
class_acc(tested_class, Y)

#WEIRD AND COMPLICATED BAYES
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
tr_data = tr_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
#loop for images 2x2, 4x4, 8x8, 16x16, 32x32
plotting_x = []
for size in [1,2,3,4,5,6,7,8,9]:
    tr_data_2x2 = cifar10_NxN_color(tr_data, size)
    X_2x2 = cifar10_NxN_color(X, size)
    # computing multivariate normal dist values of training data from all 10 classes
    muRGB_10_2x2, sigma10_2x2, p_10 = cifar10_bayes_learn(tr_data_2x2, tr_label, size)
    # classifying the images with Bayes classifier
    tested_class = []
    count = 1
    for img_index in range(len(X_2x2)):
        best_class = cifar10_classifier_bayes(X_2x2[img_index], muRGB_10_2x2, sigma10_2x2, p_10)
        tested_class.append(best_class)
    # classification accuracy for 2x2 Bayes
    print(f"Classification accuracy for {size}x{size} Bayes classifier")
    class_acc(tested_class, Y)
    plotting_x.append(class_acc(tested_class, Y))

#plotting
y = [0, 23.19, 31.73, 34.98, 36.10, 36.19, 36.56, 36.47, 35.92]
x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(x, y, color = 'black', marker = 'o')
plt.xlim(0, 10)
plt.ylim(0, 45)
plt.xlabel('Image size: N x N')
plt.ylabel('Accuracy: %')

plt.title('Classification accuracy - multivariate norm dist')
plt.show()
