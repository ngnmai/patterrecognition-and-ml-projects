import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

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

tr_data = tr_data.reshape(50000, 3072).astype("float32")
print(tr_data.shape)
tr_label = np.array(tr_label)

#classification accuracy
def class_acc(pred, gt):
    N = len(gt)
    corr = 0
    for i in range(len(gt)):
        if np.all((pred[i]) == (gt[i])):
            corr = corr + 1
    tot_correct = corr/(N)
    print(f'\n Classication accuracy: {tot_correct * 100:.2f}%')

#one hot function
def one_hot_conversion(input_array):
    output_array = np.zeros((input_array.size, input_array.max() + 1))
    output_array[np.arange(input_array.size), input_array] = 1
    return output_array

#one hot converting data
tr_label = tr_label.astype(dtype = 'int32')
tr_label = one_hot_conversion(tr_label)

#normalizing tr_data and X (test data)
tr_data, X = tr_data/255, X/255

#building layers
num_of_neurons = 100
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape = (3072)),
    tf.keras.layers.Dense(num_of_neurons, input_dim = 3072, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'tanh')
])
#learning rate
keras.optimizers.SGD(learning_rate=0.25)
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

#number of epoch
num_of_epochs = 150

model.fit(tr_data, tr_label, epochs = num_of_epochs, verbose = 1)

labels = np.array([])
label = np.array([np.squeeze(model.predict(X))])
print(label.shape)

for element in range(label.shape[1]):
    labels = np.append(labels, np.argmax(label[0][element]))
print(labels.shape)
print(labels)

print("With number of epochs = " + str(num_of_epochs))
print("Number of neurons = " + str(num_of_neurons))
print("learning rate = 0.25")
class_acc(labels, Y)
