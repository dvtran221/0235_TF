import tensorflow as tf
import numpy as np
from scipy import sparse

def activation_ReLU(x):
    return tf.maximum(0.0,x)

def softmax(Z):
    return tf.nn.softmax(Z)

def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (np.arange(len(y)),y)), shape = (len(y),C)).toarray()
    return Y

def network(X):
    n_input = 2
    n_hidden1 = 100
    n_hidden2 = 20
    n_hidden3 = 10
    n_classes = 3
    # hidden layer 1
    W1 = tf.Variable(tf.random.truncated_normal((n_input,n_hidden1)))
    b1 = tf.Variable(tf.random.truncated_normal((1,n_hidden1)))
    Z1 = tf.add(tf.matmul(X,W1),b1)
    A1 = activation_ReLU(Z1)
    tf.summary.histogram("W1",W1)
    tf.summary.histogram("b1",b1)

    # output layer
    W4 = tf.Variable(tf.random.truncated_normal((n_hidden1,n_classes)))
    b4 = tf.Variable(tf.random.truncated_normal((1,n_classes)))
    Z4 = tf.add(tf.matmul(A1,W4),b4)
    Y = softmax(Z4)

    # # hidden layer 2
    # W2 = tf.Variable(tf.random.truncated_normal((n_hidden1,n_hidden2)))
    # b2 = tf.Variable(tf.random.truncated_normal((1,n_hidden2)))
    # Z2 = tf.add(tf.matmul(A1,W2),b2)
    # A2 = activation_ReLU(Z2)
    # tf.summary.histogram("W2",W2)
    # tf.summary.histogram("b2",b2)

    # # hidden layer 3
    # W3 = tf.Variable(tf.random.truncated_normal((n_hidden2,n_hidden3)))
    # b3 = tf.Variable(tf.random.truncated_normal((1,n_hidden3)))
    # Z3 = tf.add(tf.matmul(A2,W3),b3)
    # A3 = activation_ReLU(Z3)
    # tf.summary.histogram("W3",W3)
    # tf.summary.histogram("b3",b3)

    # # output layer
    # W4 = tf.Variable(tf.random.truncated_normal((n_hidden3,n_classes)))
    # b4 = tf.Variable(tf.random.truncated_normal((1,n_classes)))
    # Z4 = tf.add(tf.matmul(A3,W4),b4)
    # Y = softmax(Z4)
    return Y


# y = np.random.multivariate_normal([1,2],[[1,2],[2,1]],100)
# print(y.shape)
# index = np.random.permutation(100)
# print(y[index[0:10],:])
