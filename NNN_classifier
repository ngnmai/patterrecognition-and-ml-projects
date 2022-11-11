import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

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

#random classifier
def cifar10_classifer_random(x):
    input_len = []
    for i in range(len(x)):
        temp_var = random.randint(0,9)
        input_len.append(temp_var)
    return input_len

#1NN classifier
def cifar10_classifer_1nn(x, trdata, trlables):
    pred_label = []
    for i in range(len(x)):
        img = x[i]
        img.astype(int)
        distances = []
        closest_distance = 0
        for n in range(trdata.shape[0]):
            compar = 0
            refimg = trdata[n]
            refimg.astype(int)
            compar = np.sum((img - refimg)**2)
            distances.append(compar)
            if n == 0:
                closest_distance = compar
                label_index = n
            else:
                if compar < closest_distance:
                    closest_distance = compar
                    label_index = n
        closest_dist_img = trdata[distances.index(closest_distance)]
        label = trlables[label_index]
        pred_label.append(label)
    return (pred_label)

labeldict = unpickle('D:\\UNI ASSIGNMENTS\\intro to pattern recog n ML\\EXERCISE 3\\cifar-10-python\\cifar-10-batches-py\\batches.meta')
label_names = labeldict["label_names"]

#storing testing data and its corresponding labels
datadict = unpickle('D:\\UNI ASSIGNMENTS\\intro to pattern recog n ML\\EXERCISE 3\\cifar-10-python\\cifar-10-batches-py\\test_batch') #testing data
X = datadict["data"] #unclassified data
Y = datadict["labels"] #ground truth label of unclassified data
X = X.astype("int32")

#storing traning data and its corresponding labels
tr_data = np.array([]) #training data
tr_label = np.array([]) #training data's labels
for i in range(1, 6):
    training_data = unpickle(f'D:\\UNI ASSIGNMENTS\\intro to pattern recog n ML\\EXERCISE 3\\cifar-10-python\\cifar-10-batches-py\\data_batch_{i}')
    tr_data = np.append(tr_data, training_data["data"])
    tr_label = np.append(tr_label, training_data["labels"])


print(X.shape)
print(len(Y))

tr_data = tr_data.reshape(50000, 3072).astype("int32")
print(tr_data.shape)
tr_label = np.array(tr_label)

print('2. CIFAR10 - EVALUATION: class_acc for predicted label pred')
class_acc(tr_label, tr_label)
print('3. CIFAR10 - RANDOM CLASSIFICATION: return random labels and test classification accuracy of them')
random_test = cifar10_classifer_random(tr_label)
class_acc(random_test, tr_label)
print('4. CIFAR10 - 1NN CLASSIFIER')
pred_1NN = cifar10_classifer_1nn(X, tr_data, tr_label)
class_acc(pred_1NN, Y)
