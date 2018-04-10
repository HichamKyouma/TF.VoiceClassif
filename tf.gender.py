import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf
import random

dataset=pd.read_csv('path to voice.csv')
X=dataset.iloc[:,0:20]
y=dataset.iloc[:,-1].values

#assign male=1 & female=0 

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
print(y)

#scaling 
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X= X_sc.fit_transform(X)

#shuffling the data
def shuf(X,Y):
    tr=[]
    ts=[]
    r= random.sample(range(len(X)), len(X))   
    for i in r:
        tr.append(X[i])
        ts.append(Y[i])
    return tr,ts
    
X,y = shuf(X,y)

#data = trainingdata + testdata

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Neural Network Model

def CNN(input_data):
    
    input_layer=tf.reshape(input_data, [-1,20])
    
    HL1 = tf.layers.dense(inputs=input_data, units=100, activation = tf.nn.relu)
    
    HL2 = tf.layers.dense(inputs=HL1, units=100, activation = tf.nn.relu)
    
    HL3 = tf.layers.dense(inputs = HL2, units = 55, activation = tf.nn.relu)
    
    logit =  tf.layers.dense(inputs = HL2, units = 2, activation= tf.nn.sigmoid)
    
    return logit
#define placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 20])
y = tf.placeholder(dtype = tf.int32, shape = [None])

logits = CNN(x)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#start training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(70):
        _, loss_val = sess.run([train_op, loss], feed_dict={x: X_train, y: y_train})
        print('EPOCH',i, 'loss :', loss_val)
        print('DONE WITH EPOCH\n')
        
#tests       
sample_indexes = random.sample(range(len(X_test)), 10)
sample_sounds = [X_test[i] for i in sample_indexes]
sample_labels = [y_test[i] for i in sample_indexes]

#comparing a sample of 10 test elements
samples = np.asarray(sample_labels)
print(samples)
print(predicted)

#Accuracy Calculation
correct_pred = tf.argmax(logits, 1)
predicted = sess.run([correct_pred], feed_dict={x: sample_sounds})[0]
predicted = sess.run([correct_pred], feed_dict={x: X_test})[0]
match_count = sum([int(y==y_) for y,y_ in zip(y_test, predicted)])
precision= match_count/len(y_test)
print("accuracy:{:.3f}".format(precision))