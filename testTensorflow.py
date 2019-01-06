import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Simple demo of classification using neural network with tf.Variables

# Step 1:  create train data and validation data
np.random.seed(10)
N = 200 
N_val = 25
n_input = 2
n_classes = 3

X_train = np.zeros((N*n_classes,n_input)) # data matrix (each row = single example)
y_train = np.zeros((N*n_classes,), dtype='uint8') # class labels
X_val = np.zeros((N_val*n_classes,n_input)) # data matrix (each row = single example)
y_val = np.zeros((N_val*n_classes,), dtype='uint8') # class labels

for j in range(n_classes):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X_train[ix,:] = np.c_[r*np.sin(t), r*np.cos(t)]
    y_train[ix] = j

for j in range(n_classes):
    ix = range(N_val*j,N_val*(j+1))
    r = np.linspace(0.0,1,N_val) # radius
    t = np.linspace(j*4,(j+1)*4,N_val) + np.random.randn(N_val)*0.25 # theta
    X_val[ix,:] = np.c_[r*np.sin(t), r*np.cos(t)]
    y_val[ix] = j

y_train = utils.convert_labels(y_train,n_classes)
y_val = utils.convert_labels(y_val,n_classes)

plt.plot(X_train[:N,0], X_train[:N,1], 'bs', markersize = 7)
plt.plot(X_train[N:2*N,0], X_train[N:2*N,1], 'ro', markersize = 7)
plt.plot(X_train[2*N:,0], X_train[2*N:,1], 'g^', markersize = 7)

plt.plot(X_val[:N_val,0], X_val[:N_val,1], 'ks', markersize = 7)
plt.plot(X_val[N_val:2*N_val,0], X_val[N_val:2*N_val,1], 'ko', markersize = 7)
plt.plot(X_val[2*N_val:,0], X_val[2*N_val:,1], 'k^', markersize = 7)

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.show()
# means = [[1, 2], [7, 2], [6,5]]
# cov = [[.5, .3], [.3, .5]]
# X0 = np.random.multivariate_normal(means[0],cov,N)
# X1 = np.random.multivariate_normal(means[1],cov,N)
# X2 = np.random.multivariate_normal(means[2],cov,N)
# # Y label is represented in one-hot coding
# X_train = np.concatenate((X0[0:175,:],X1[0:175,:],X2[0:175,:]),axis=0)
# y_train = np.concatenate((0*np.ones((1,175)),1*np.ones((1,175)),2*np.ones((1,175))),axis = 1)
# y_train = tf.one_hot(y_train,3)
# X_val = np.concatenate((X0[175:,:],X1[175:,:],X2[175:,:]),axis=0)
# y_val = np.concatenate((0*np.ones((1,25)),1*np.ones((1,25)),2*np.ones((1,25))),axis = 1)
# y_val = tf.one_hot(y_val,3)
# y_train = y_train[-1,:]
# y_val = y_val[-1,:]

# Step 2: construct a network model in tensorflow (TF) (graph)

learning_rate = 1
n_epoch = 10000
N_train = X_train.shape[0]
batch = N_train
N_iter = np.round(N_train/batch).astype(np.int32)


X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
Y_network = utils.network(X)
losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=Y_network))
optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(losses)
# define accuracy
correct_pred = tf.equal(tf.argmax(Y_network, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar("train accuracy",accuracy)
tf.summary.scalar("train loss",losses)

val_acc = tf.summary.scalar("validation accuracy",accuracy)
val_loss = tf.summary.scalar("validation loss",losses)

# Step 3: Run the training process with TF
# Don't forget global
init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph1',graph=tf.get_default_graph())
    train_summary = tf.summary.merge_all()
    val_summay = tf.summary.merge([val_acc,val_loss])
    sess.run(init)
    total_iter = 0
    for e_iter in range(n_epoch):
        #index = np.random.permutation(N_train)
        index = np.arange(N_train)
        for iter in range(N_iter):
            total_iter += 1
            idx = index[iter*batch:np.minimum(N_train,(iter+1)*batch)]
            x_batch = X_train[idx,:]
            y_batch = y_train[idx,:]
            train_loss, train_acc, _ = sess.run([losses,accuracy,optim],feed_dict={X:x_batch,Y:y_batch})
            # sess.run(optim,feed_dict={X:x_batch,Y:y_batch})
            # train_loss, train_acc = sess.run([losses,accuracy],feed_dict={X:x_batch,Y:y_batch})
            train_sum = sess.run(train_summary,feed_dict={X:x_batch,Y:y_batch})
            val_sum = sess.run(val_summay,feed_dict={X:X_val,Y:y_val})
            writer.add_summary(train_sum, global_step=total_iter)
            writer.add_summary(val_sum, global_step=total_iter)

            if total_iter%100 ==0:
                print("Epoch: {}, iteration: {}, training loss: {:.4f}, training accuracy: {:.4f}".format(e_iter,iter,train_loss,train_acc))
            
            # if iter%50 ==0:
            #     val_loss, val_acc = sess.run([losses,accuracy],feed_dict={X:X_val,Y:y_val})
            #     print("Epoch: {}, iteration: {}, validation loss: {:.4f}, validation accuracy: {:.4f}".format(e_iter,iter, train_loss,train_acc))
            writer.flush()


# Step 4: If the designed network gives a acceptable performance, then test on some real data...


