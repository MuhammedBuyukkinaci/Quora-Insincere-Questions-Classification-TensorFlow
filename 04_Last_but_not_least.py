#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import os
from tqdm import tqdm
import re
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[2]:


tqdm.pandas('desc')


# In[3]:


#Always seed the randomness of this universe.
SEED = 51
np.random.seed(SEED)


# In[4]:


#Define HyperParameters
scoring_amount = 512 # how many samples taken to score test data
n_splits = 5 # how many folds
MAX_WORD_TO_USE = 150000 # how many words to use in training
MAX_LEN = 72 # number of time-steps.
EMBED_SIZE = 300 #GLoVe 100-D
batchSize = 512 # how many samples to feed neural network
GRU_UNITS = 256 # Number of nodes in GRU Layer
numClasses = 2 #{Sincere,Insincere}
attention_size = 64 # how many nodes in attention layer
iterations = 10000 # How many iterations to train
nodes_on_FC = 64 # Number of nodes on FC layer
epsilon = 1e-4# For batch normalization
val_loop_iter = 200 # in how many iters we record
SEED = 51 # our seed


# In[5]:


#Reading csv's
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[6]:


train['question_text'] = train['question_text'].progress_apply(lambda x:str(x))
test['question_text'] = test['question_text'].progress_apply(lambda x:str(x))


# In[7]:


train.head()


# In[8]:


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text
def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")    


# In[9]:


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


# In[10]:


# lower
train["question_text"] = train["question_text"].apply(lambda x: x.lower())
test["question_text"] = test["question_text"].apply(lambda x: x.lower())

# Clean the text
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))

# Clean numbers
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_numbers(x))

# Clean speelings
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
test["question_text"] = test["question_text"].apply(lambda x: replace_typical_misspell(x))

## fill up the missing values
train_X = train["question_text"].fillna("_##_").values
test_X = test["question_text"].fillna("_##_").values


# In[11]:


## Tokenize the sentences
tokenizer = Tokenizer(num_words=MAX_WORD_TO_USE)
tokenizer.fit_on_texts(list(train['question_text']) )
train_X_temp = tokenizer.texts_to_sequences(train['question_text'])
test_X_temp = tokenizer.texts_to_sequences(test['question_text'])


# In[12]:


## Pad the sentences 
train_X = pad_sequences(train_X_temp, maxlen=MAX_LEN,truncating = 'post',padding='pre')
test_X = pad_sequences(test_X_temp, maxlen=MAX_LEN)


# In[13]:


train_y = train['target'].values


# In[14]:


word_index = tokenizer.word_index


# In[15]:


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_WORD_TO_USE, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in tqdm(word_index.items()):
        if i >= MAX_WORD_TO_USE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 


# In[16]:


def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_WORD_TO_USE, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_WORD_TO_USE: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


# In[18]:


glove_embeddings = load_glove(word_index)
paragram_embeddings = load_para(word_index)


# # CUDNNGRU, ATTENTION , GLOVE , ADAM

# In[20]:


embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
embedding_matrix = embedding_matrix.astype('float32')
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(train_X, train_y))


# In[23]:


#Resetting the graph
tf.reset_default_graph()

#Seed the randomness
tf.set_random_seed(SEED)

#Defining Placeholders
input_data = tf.placeholder(tf.int32, [None, MAX_LEN])
y_true = tf.placeholder(tf.float32, [None, numClasses])

hold_prob1 = tf.placeholder(tf.float32)
#Creating our Embedding matrix
data = tf.nn.embedding_lookup(embedding_matrix,input_data)

data = tf.transpose(data, [1, 0, 2])

#For single layer GRU
GRU_CELL = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1,num_units=GRU_UNITS,                                         bias_initializer = tf.constant_initializer(0.1),                                         kernel_initializer=tf.contrib.layers.xavier_initializer() )

value, _ = GRU_CELL(inputs= data)

last = tf.gather(value, int(value.get_shape()[0]) - 1)

##Dropout Layer 1
print(last.get_shape().as_list())

y_pred_with_dropout = tf.nn.dropout(last,keep_prob=hold_prob1)

#Defining weights and biases for 1 st Fully Connected part of NN
weight_output_layer = tf.Variable(tf.truncated_normal([GRU_UNITS, numClasses]))
bias_output_layer = tf.Variable(tf.constant(0.1, shape=[numClasses]))
#Calculating last layer of NN, without any activation
y_pred = tf.matmul(y_pred_with_dropout, weight_output_layer) + bias_output_layer

y_pred_softmax = tf.nn.softmax(y_pred)

#Defining Accuracy
matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

#Defining Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
#Defining objective
training = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

##Initializing trainable/non-trainable variables
init = tf.global_variables_initializer()

print("Model was built up")


# In[24]:


train_preds = np.zeros((len(train_X),2))
test_preds = []
#GPU settings
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
#Opening up Session
for j, (train_idx, valid_idx) in enumerate(splits): 
    print( "Fold {} has started".format(j) )
    with tf.Session(config=config) as sess:
        #Running init
        sess.run(init)    
        
        #Main loop
        for i in range(iterations):
            random_numbers = np.random.choice(train_idx,batchSize)
            _,c = sess.run([training,cross_entropy] ,feed_dict = {input_data : train_X[random_numbers],            y_true : pd.get_dummies(train_y[random_numbers]).values, hold_prob1:0.7} )

            #Validating Loop
            if i % val_loop_iter == 0:
                random_numbers_cv = np.random.choice(valid_idx,batchSize)
                
                #Getting train stats.
                acc_tr,loss_tr = sess.run([acc,cross_entropy],                feed_dict={input_data:train_X[random_numbers],                           y_true:pd.get_dummies(train_y[random_numbers]).values, hold_prob1:1.0 })
                
                #Getting validation stats.
                acc_cv,loss_cv = sess.run([acc,cross_entropy],                feed_dict = {input_data:train_X[random_numbers_cv],                             y_true:pd.get_dummies(train_y[random_numbers_cv]).values,hold_prob1:1.0})
                
                #Printing on each 1000 iterations
                if i%1000 ==0:
                    print("Training  : Iter = {}, Train Loss = {}, Train Accuracy = {}".format(i,loss_tr,acc_tr))
                    print("Validation: Iter = {}, CV    Loss = {}, CV Accuracy = {}".format(i,loss_cv,acc_cv))
        
        #OOF predictions        
        for r in range(0,len(valid_idx),scoring_amount):
            index_train = valid_idx[r:r + scoring_amount]
            train_preds[index_train,:] = y_pred_softmax.eval(feed_dict={input_data:train_X[index_train],hold_prob1:1.0},session=sess)

        #Test predictions        
        test_fold = []
        for r in range(0,len(test_X),scoring_amount):
            k = y_pred_softmax.eval(feed_dict={input_data:test_X[r:r+scoring_amount],hold_prob1:1.0},session=sess)
            test_fold.append(k)
            del k
        test_preds.append(test_fold)
        
        print("Training {} has finished".format(j))


# In[28]:


our_preds_test = []
for i in test_preds:
    l = np.zeros((0,2))
    for j in i:
        l = np.append(l,j,axis=0)
    our_preds_test.append(l)
test_predictions = our_preds_test[0] + our_preds_test[1] + our_preds_test[2] + our_preds_test[3] + our_preds_test[4]
test_predictions = test_predictions / 5
test_predictions = test_predictions[:,1]


# # CUDNNLSTM, ATTENTION, GLOVE + FASTTEXT , RMSPROP

# In[29]:


#Resetting the graph
tf.reset_default_graph()

#Seed the randomness
tf.set_random_seed(SEED)

#Defining Placeholders
input_data = tf.placeholder(tf.int32, [None, MAX_LEN])
y_true = tf.placeholder(tf.float32, [None, numClasses])

hold_prob1 = tf.placeholder(tf.float32)
#Creating our Embedding matrix
data = tf.nn.embedding_lookup(embedding_matrix,input_data)

data = tf.transpose(data, [1, 0, 2])

#For single layer GRU
GRU_CELL = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,num_units=GRU_UNITS,                                         bias_initializer = tf.constant_initializer(0.1),                                         kernel_initializer=tf.contrib.layers.xavier_initializer() )

value, _ = GRU_CELL(inputs= data)

last = tf.gather(value, int(value.get_shape()[0]) - 1)

#Defining weights and biases for 1 st Fully Connected part of NN
weight_fc1 = tf.Variable(tf.truncated_normal([GRU_UNITS, nodes_on_FC]))
bias_fc1 = tf.Variable(tf.constant(0.1, shape=[nodes_on_FC]))

#Defining 1st FC layer
y_pred_without_BN = tf.matmul(last, weight_fc1) + bias_fc1
#calculating batch_mean and batch_variance
batch_mean, batch_var = tf.nn.moments(y_pred_without_BN,[0])
#Creating parameters for Batch normalization
scale = tf.Variable(tf.ones([nodes_on_FC]))
beta = tf.Variable(tf.zeros([nodes_on_FC]))
#Implementing batch normalization
y_pred_without_activation = tf.nn.batch_normalization(y_pred_without_BN,batch_mean,batch_var,beta,scale,epsilon)

#Applying RELU
y_pred_with_activation = tf.nn.relu(y_pred_without_activation)
#Dropout Layer 1
y_pred_with_dropout = tf.nn.dropout(y_pred_with_activation,keep_prob=hold_prob1)

#Defining weights and biases for 1 st Fully Connected part of NN
weight_output_layer = tf.Variable(tf.truncated_normal([nodes_on_FC, numClasses]))
bias_output_layer = tf.Variable(tf.constant(0.1, shape=[numClasses]))
#Calculating last layer of NN, without any activation
y_pred = tf.matmul(y_pred_with_dropout, weight_output_layer) + bias_output_layer

y_pred_softmax = tf.nn.softmax(y_pred)


#Defining Accuracy
matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

#Defining Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
#Defining objective
training = tf.train.RMSPropOptimizer(learning_rate=0.0003).minimize(cross_entropy)

##Initializing trainable/non-trainable variables
init = tf.global_variables_initializer()

print("Model was built up")


# In[30]:


train_preds2 = np.zeros((len(train_X),2))
test_preds2 = []
#GPU settings
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
#Opening up Session
for j, (train_idx, valid_idx) in enumerate(splits): 
    print( "Fold {} has started".format(j) )
    with tf.Session(config=config) as sess:
        #Running init
        sess.run(init)    
        #Main loop
        for i in range(iterations):
            random_numbers = np.random.choice(train_idx,batchSize)
            _,c = sess.run([training,cross_entropy] ,feed_dict = {input_data : train_X[random_numbers],\
            y_true : pd.get_dummies(train_y[random_numbers]).values, hold_prob1:0.7} )

            #Validating Loop
            if i % val_loop_iter == 0:
                random_numbers_cv = np.random.choice(valid_idx,batchSize)
                
                #Getting train stats.
                acc_tr,loss_tr = sess.run([acc,cross_entropy],feed_dict={input_data:train_X[random_numbers],\
                y_true:pd.get_dummies(train_y[random_numbers]).values, hold_prob1:1.0 })
                                
                #Getting validation stats.
                acc_cv,loss_cv = sess.run([acc,cross_entropy],feed_dict = {input_data:train_X[random_numbers_cv],y_true:pd.get_dummies(train_y[random_numbers_cv]).values,hold_prob1:1.0})
                
                #Printing on each 1000 iterations
                if i%1000 ==0:
                    print("Training  : Iter = {}, Train Loss = {}, Train Accuracy = {}".format(i,loss_tr,acc_tr))
                    print("Validation: Iter = {}, CV    Loss = {}, CV Accuracy = {}".format(i,loss_cv,acc_cv))
        
        #OOF predictions        
        for r in range(0,len(valid_idx),scoring_amount):
            index_train = valid_idx[r:r + scoring_amount]
            train_preds2[index_train,:] = y_pred_softmax.eval(feed_dict={input_data:train_X[index_train],hold_prob1:1.0},session=sess)

        #Test predictions        
        test_fold = []
        for r in range(0,len(test_X),scoring_amount):
            k = y_pred_softmax.eval(feed_dict={input_data:test_X[r:r+scoring_amount],hold_prob1:1.0},session=sess)
            test_fold.append(k)
            del k
        test_preds2.append(test_fold)
        print("Training {} has finished".format(j))
        sess.close()


# In[31]:


our_preds_test2 = []
for i in test_preds2:
    l = np.zeros((0,2))
    for j in i:
        l = np.append(l,j,axis=0)
    our_preds_test2.append(l)
test_predictions2 = our_preds_test2[0] + our_preds_test2[1] + our_preds_test2[2] + our_preds_test2[3] + our_preds_test2[4]
test_predictions2 = test_predictions2 / 5
test_predictions2 = test_predictions2[:,1]


# # PREDICTING

# In[32]:


training_predictions = pd.concat([ pd.Series(train_preds[:,1]), pd.Series(train_preds2[:,1])],axis=1)
testing_predictions = pd.concat([ pd.Series(test_predictions), pd.Series(test_predictions2) ],axis=1)


# In[33]:


training_predictions.columns = ['model_1','model_2']
testing_predictions.columns = ['model_1','model_2']


# In[34]:


inverse_w_model1 = sqrt(mean_squared_error(train_y, training_predictions['model_1']))
inverse_w_model2 = sqrt(mean_squared_error(train_y, training_predictions['model_2']))
inverse_w_model1 = 1 / inverse_w_model1
inverse_w_model2 = 1 / inverse_w_model2


# In[35]:


w1 = (inverse_w_model1) / (inverse_w_model1+ inverse_w_model2)
w2 = (inverse_w_model2) / (inverse_w_model1+ inverse_w_model2)


# In[36]:


print(w1,w2)


# In[48]:


training_predictions['model_1_pred'] = training_predictions['model_1'].apply(lambda x: 1 if x > 0.30 else 0)
training_predictions['model_2_pred'] = training_predictions['model_2'].apply(lambda x: 1 if x > 0.30 else 0)


# In[49]:


print("Model1 F1=")
print(f1_score(train_y,training_predictions['model_1_pred'].values))


# In[50]:


print("Model2 F1=")
print(f1_score(train_y,training_predictions['model_2_pred'].values))


# In[51]:


training_predictions['merged']=training_predictions['model_1']*w1 + training_predictions['model_2']*w2
testing_predictions['merged']=testing_predictions['model_1']*w1 + testing_predictions['model_2']*w2
training_predictions['prediction'] = training_predictions['merged'].apply(lambda x: 1 if x > 0.300 else 0)
testing_predictions['prediction'] = testing_predictions['merged'].apply(lambda x: 1 if x > 0.300 else 0)


# In[52]:


print("Train F1 = ")
print(f1_score(train_y,training_predictions['prediction'].values))


# In[53]:


training_predictions.head()


# In[54]:


submission_file = pd.concat([test['qid'],testing_predictions['prediction']],axis=1)


# In[55]:


submission_file.head()

submission_file.to_csv('submission.csv', index=False)

