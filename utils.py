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
    n_hidden1 = 50
    n_hidden2 = 50
    n_hidden3 = 10
    n_classes = 3
    # hidden layer 1
    with tf.variable_scope("Hidden_layer_1"):
        W1 = tf.Variable(tf.random_normal((n_input,n_hidden1))*tf.sqrt(2.0/n_input), name = 'W1')
        b1 = tf.Variable(tf.zeros((1,n_hidden1)), name = 'b1')
        Z1 = tf.add(tf.matmul(X,W1),b1)
        A1 = activation_ReLU(Z1)
        tf.summary.histogram("W1",W1)
        tf.summary.histogram("b1",b1)    

    # hidden layer 2
    with tf.variable_scope("Hidden_layer_2"):
        W2 = tf.Variable(tf.random_normal((n_hidden1,n_hidden2))*tf.sqrt(2.0/n_hidden1), name = 'W2')
        b2 = tf.Variable(tf.zeros((1,n_hidden2)), name = 'b2')
        Z2 = tf.add(tf.matmul(A1,W2),b2)
        A2 = activation_ReLU(Z2)
        tf.summary.histogram("W2",W2)
        tf.summary.histogram("b2",b2)

    
    # # hidden layer 3
    # with tf.variable_scope("Hidden_layer_3"):
    # W3 = tf.Variable(tf.random_normal((n_hidden2,n_hidden3))*tf.sqrt(2.0/n_hidden2), name = 'W3')
    # b3 = tf.Variable(tf.zeros((1,n_hidden3)), name = 'b3')
    # Z3 = tf.add(tf.matmul(A2,W3),b3)
    # A3 = activation_ReLU(Z3)
    # tf.summary.histogram("W3",W3)
    # tf.summary.histogram("b3",b3)

    # output layer
    with tf.variable_scope("Output_layer"):
        W4 = tf.Variable(tf.random_normal((n_hidden2,n_classes))*tf.sqrt(2.0/n_hidden2), name = 'W4')
        b4 = tf.Variable(tf.zeros((1,n_classes)), name = 'b4')
        Z4 = tf.add(tf.matmul(A2,W4),b4)
        Y = softmax(Z4)
    return Y

def prediction(X, W, B):
    Z1 = tf.add(tf.matmul(X,W['Hidden_layer_1']),B['Hidden_layer_1'])
    A1 = activation_ReLU(Z1)
    Z2 = tf.add(tf.matmul(A1,W['Hidden_layer_2']),B['Hidden_layer_2'])
    A2 = activation_ReLU(Z2)
    Z3 = tf.add(tf.matmul(A2,W['Output_layer']),B['Output_layer'])
    Y = softmax(Z3)
    return tf.argmax(Y, axis = 1)
