
# coding: utf-8

# In[1]:

import read_20newsgroup_stemmed as ng
import tensorflow as tf
from gensim.models import KeyedVectors
import numpy as np


# In[2]:

#to read the data
print("Reading the data.....")
data = ng.ReadNewsGroup('./data/', rare_threshold=5, seed=10)


# In[ ]:


##pretrained word2vec model
print("loading the word2vec model...")
word_vec = KeyedVectors.load_word2vec_format('./data/wiki.en.vec')


# In[ ]:

#parameters
batch_size = 50
max_length = 430
word_vec_size = 300
num_batches = 1000
num_classes = 20


# In[27]:

# Define the function from words to vectors
def word2vec(sentences):
    n = len(sentences)
    word_vecs = np.zeros((batch_size, max_length, word_vec_size))
    skipped = 0
    total = 0
    for i in range(n):
        s = sentences[i]
        m = len(s)
        total += m
        k = 0
        for j in range(max_length - m, max_length):
            try:
                word_vecs[i, j, :] = word_vec[s[k]]
            except:
                skipped += 1
            k += 1
    #to count miss word in vocabulary
    #print('Skipped', skipped, 'out of', total)
    return word_vecs

print("training the model...")
# In[28]:

try:
    del lstm_size
    del lstm
    del hidden_state
    del current_state
    del words
    del state
    del output
    tf.reset_default_graph()
except:
    pass
    
lstm_size = 200
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
hidden_state = tf.zeros([batch_size, lstm_size])
current_state = tf.zeros([batch_size, lstm_size])
state = hidden_state, current_state

words = tf.placeholder(tf.float32, [batch_size, max_length, word_vec_size])
labels = tf.placeholder(tf.float32, [batch_size, num_classes])

output = tf.zeros([batch_size, lstm_size])
with tf.variable_scope("RNN"):
    for i in range(max_length):
        if i > 0: tf.get_variable_scope().reuse_variables()
        output, state = lstm(tf.reshape(words[:, i, :],                                         [batch_size, -1]), state)

output = tf.layers.dense(output, num_classes)
pred = tf.nn.softmax(output)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(                    logits=output, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss_op)


# In[29]:

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[30]:

for i in range(num_batches):
    e, l = data.get_data(batch_size, set_type='train')
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        email = e[i]
        label = l[i]
        if len(email) <= max_length:
            batch_x.append(email)
            batch_y.append(label)
            
    batch_x = word2vec(batch_x)
    batch_y = np.asarray(batch_y)
    temp = np.zeros((batch_size, num_classes))
    n = batch_y.shape[0]
    for i in range(n):
        temp[i, batch_y[i]] = 1
    for i in range(n, batch_size):
        temp[i, 0] = 1
    batch_y = temp
    
    _, cost = sess.run([train_op, loss_op], feed_dict={words: batch_x,                                                    labels: batch_y})
    
    print(cost)
    


# In[23]:

for i in range(num_batches):
    sum=0
    e, l = data.get_data(batch_size, set_type='test')
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        email = e[i]
        label = l[i]
        if len(email) <= max_length:
            batch_x.append(email)
            batch_y.append(label)
            
    batch_x = word2vec(batch_x)
    batch_y = np.asarray(batch_y)
    temp = np.zeros((batch_size, num_classes))
    n = batch_y.shape[0]
    for i in range(n):
        temp[i, batch_y[i]] = 1
    for i in range(n, batch_size):
        temp[i, 0] = 1
    batch_y = temp
    p= sess.run(pred, feed_dict={words: batch_x})
    idx=np.argmax(p,axis=1)
    true_idx=np.argmax(batch_y,axis=1)
    acc=np.mean(idx==true_idx)
    sum=sum+acc


# In[25]:

total_acc=sum/1000
print('test set accuracy-',total_acc)


# In[22]:


#for checking accuracy on training data
""""
for i in range(num_batches):
    e, l = data.get_data(batch_size, set_type='train')
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        email = e[i]
        label = l[i]
        if len(email) <= max_length:
            batch_x.append(email)
            batch_y.append(label)
            
    batch_x = word2vec(batch_x)
    batch_y = np.asarray(batch_y)
    temp = np.zeros((batch_size, num_classes))
    n = batch_y.shape[0]
    for i in range(n):
        temp[i, batch_y[i]] = 1
    for i in range(n, batch_size):
        temp[i, 0] = 1
    batch_y = temp
    p= sess.run(pred, feed_dict={words: batch_x})
    idx=np.argmax(p,axis=1)
    true_idx=np.argmax(batch_y,axis=1)
    print(np.mean(idx==true_idx))
"""


# In[ ]:



