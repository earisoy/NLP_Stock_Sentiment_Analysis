# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:44:39 2021

@author: erdem.arisoy
"""

# import key libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# NLTK
import nltk
from nltk.corpus import stopwords
#Gensim
import gensim

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D,Conv2D, MaxPool1D, Bidirectional, Dropout
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
# Sub libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.compose import make_column_selector


import os
import sys
import datetime as dt
import joblib
import json
import string
import re


#%% Define paraöeters
try:
    
    #vocab_size = 1000  # to be able adjust # of word to use
    embedding_dim = 512 #Embedding dimension
    max_length = 20 # Max word used in tokenize part 
    trunc_type='post' # Truncate long sentences
    padding_type='post' # Padding short sentence
    oov_tok = "<OOV>" #The word that replaced new words in test dataset
    training_portion = 0.8 #Training size
    
    
    datetime_object = str(dt.datetime.now().strftime("%Y%m%d%H%M%S"))
    print(datetime_object)

    director_parameter = 'C:\\Users\\erdem.arisoy\\Desktop\\NLP'
    os.chdir(director_parameter)
    
    log_file_path = director_parameter + "\\" +'log-file-train-'+datetime_object+".log"
    f = open(log_file_path, 'w')
    
    f.write(str(dt.datetime.now())+"___"+"Logging Started For NLP model_:"+'\n')
    print(str(dt.datetime.now())+"___"+"Logging Started For NLP model:"'\n')

#%% Import data

   # load the data
    stock_df = pd.read_csv("stock_sentiment.csv") 
    stock_df.head()
    
    f.write(str(dt.datetime.now())+"___"+"Step0_Data loading done\n")
    print(str(dt.datetime.now())+"___"+"Step0_Data loading done\n")
    
    
    print(str(dt.datetime.now())+"___"+"Step1 # of row: "+str(stock_df['Sentiment'].count())+'\n')
    print(str(dt.datetime.now())+"___"+"Step1 # of 1: "+str(stock_df['Sentiment'].sum())+'\n')
    f.write(str(dt.datetime.now())+"___"+"Step1 # of row: "+str(stock_df['Sentiment'].count())+'\n')
    f.write(str(dt.datetime.now())+"___"+"Step1 # of 1: "+str(stock_df['Sentiment'].sum())+'\n')

#%% Remove punctiotion    

    def remove_punc(message):
        Test_punc_removed = [char for char in message if char not in string.punctuation]
        Test_punc_removed_join = ''.join(Test_punc_removed)
    
        return Test_punc_removed_join
    
    # Remove punctuations from our dataset 
    stock_df['Text'] = stock_df['Text'].apply(remove_punc)
   
    print(str(dt.datetime.now())+"___"+"Step2 punctiotions are removed" +'\n')
    f.write(str(dt.datetime.now())+"___"+"Step2 Punctiotions are removed" +'\n')
        
#%% Remove stopwords
       
    # download stopwords
    nltk.download("stopwords")
    stopwords.words('english')
    
    # Obtain additional stopwords from nltk
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['https','from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year', 'https', 'and'])
 

# Remove stopwords and remove short words (less than 2 characters)
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if len(token) >= 3 and token not in stop_words:
                result.append(token)
                
        return result

    # apply pre-processing to the text column
    stock_df['Text'] = stock_df['Text'].apply(preprocess)
    
    print(str(dt.datetime.now())+"___"+"Step2 stopwords are removed" +'\n')
    f.write(str(dt.datetime.now())+"___"+"Step2 stopwords are removed" +'\n')

#%% Obtain total number of the words

# Obtain the total words present in the dataset
    list_of_words = []
    for i in stock_df['Text']:
        for j in i:
            list_of_words.append(j)
    # Obtain the total number of unique words
    total_words = len(list(set(list_of_words)))

    print(str(dt.datetime.now())+"___"+"Step3 total numbers of words"+str(total_words)+'\n')
    f.write(str(dt.datetime.now())+"___"+"Step3 total numbers of words"+str(total_words)+'\n')

#%% Train- test split


    train_sentences, test_sentences, train_labels, test_labels = train_test_split(stock_df['Text'], stock_df['Sentiment'], test_size =1-training_portion)

    print(str(dt.datetime.now())+"___"+"Step4__"+"Train data shape :"+ str(train_sentences.shape))
    print(str(dt.datetime.now())+"___"+"Step4__"+"Test data shape :"+str(test_sentences.shape))     
    
    print(str(dt.datetime.now())+"___"+"Step4__"+'# of 1 in the target (train):'+str(train_labels[train_labels==1].count()))
    print(str(dt.datetime.now())+"___"+"Step4__"+'# of 0 in the target (train):'+str(train_labels[train_labels==0].count()))
    
    print(str(dt.datetime.now())+"___"+"Step4__"+'# of 1 in the target (test):'+str(test_labels[test_labels==1].count()))
    print(str(dt.datetime.now())+"___"+"Step4__"+'# of 0 in the target (test):'+str(test_labels[test_labels==0].count()))
    
    f.write(str(dt.datetime.now())+"___"+"Step4__"+"Train data shape :"+ str(train_sentences.shape)+'\n')
    f.write(str(dt.datetime.now())+"___"+"Step4__"+"Test data shape :"+str(test_sentences.shape)+'\n')     
    
    f.write(str(dt.datetime.now())+"___"+"Step4__"+'# of 1 in the target (train):'+str(train_labels[train_labels==1].count())+'\n')
    f.write(str(dt.datetime.now())+"___"+"Step4__"+'# of 0 in the target (train):'+str(train_labels[train_labels==0].count())+'\n')
    
    
    f.write(str(dt.datetime.now())+"___"+"Step4__"+'# of 1 in the target (test):'+str(test_labels[test_labels==1].count())+'\n')
    f.write(str(dt.datetime.now())+"___"+"Step4__"+'# of 0 in the target (test):'+str(test_labels[test_labels==0].count())+'\n')    
    
    train_labels=to_categorical(train_labels, 2)
    test_labels=to_categorical(test_labels, 2)
    
#%% Tokenize train and test data

    tokenizer = Tokenizer(num_words = total_words, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, truncating = trunc_type, maxlen=max_length)
    
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences,padding=padding_type,truncating = trunc_type,maxlen=max_length)
    
    print(str(dt.datetime.now())+"___"+"Step5__"+"padded train data shape :"+ str(train_padded.shape))
    print(str(dt.datetime.now())+"___"+"Step5__"+"padded test data shape :"+ str(test_padded.shape))

    f.write(str(dt.datetime.now())+"___"+"Step5__"+"padded train data shape :"+ str(train_padded.shape))
    f.write(str(dt.datetime.now())+"___"+"Step5__"+"padded test data shape :"+ str(test_padded.shape))

    # Delete unused tables
    del(stock_df,train_sentences,train_sequences,test_sentences,test_sequences)    


#%% Building Neural network model
 
    model = Sequential()
    model.add(Embedding(total_words, output_dim = embedding_dim,input_length=20))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,5,activation='relu'))
    model.add(MaxPool1D(pool_size =4))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.1)) )
    model.add(Dropout(0.3))
    model.add(Dense(2,activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
    

    
    print(model.summary())
    f.write(str(model.summary()))

    # Create callback for early stopping
    es =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
    
    #Fit the model
    history=model.fit(train_padded, train_labels, 
                      batch_size = 32, 
                      validation_split = 0.2,
                      epochs = 100,
                      callbacks=es)

    
#%% Plot train and validation performance history
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs=range(len(acc)) # Get number of epochs
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    
    plt.figure()
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])

    plt.figure()
    


#%% Performance metrics

    #-----------------------------------------------------------------------------#
    #Best Pipeline train performance metrics:
    
    
    y_train_prob1 = model.predict(train_padded)[:,1]
    y_train_predict = np.argmax(model.predict(train_padded), axis=-1)

    train_f1 = np.round(f1_score(train_labels[:,1], y_train_predict),3)   
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train f1 Score: " + str(train_f1) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train f1 Score: " + str(train_f1) +"\n")
    
    train_recall = np.round(recall_score(train_labels[:,1], y_train_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Recall: " + str(train_recall) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Recall: " + str(train_recall) +"\n")
    
    train_precision = np.round(precision_score(train_labels[:,1], y_train_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Precision: " + str(train_precision) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Precision: " + str(train_precision) +"\n")
    
    train_accuracy = np.round(accuracy_score(train_labels[:,1], y_train_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Accuracy: " + str(train_accuracy) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Accuracy: " + str(train_accuracy) +"\n")
    
    train_roc = np.round(roc_auc_score(train_labels[:,1], y_train_prob1),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train ROC: " + str(train_roc) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train ROC: " + str(train_roc) +"\n")
    
    train_lift =np.round( (precision_score(train_labels[:,1], y_train_predict)/ (train_labels[:,1].sum()/len(train_labels[:,1]))) ,3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Lift: " + str(train_lift) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Train Lift: " + str(train_lift) +"\n")
    
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_labels[:,1], y_train_predict).ravel()
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline Train "+ " tn: " + str(tn_train) + " fp: " + str(fp_train) + " fn: " + str(fn_train) + " tp: " + str(tp_train) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline Train "+ " tn: " + str(tn_train) + " fp: " + str(fp_train) + " fn: " + str(fn_train) + " tp: " + str(tp_train) +"\n")
    #-----------------------------------------------------------------------------#
    #Best Pipeline test performance metrics:
    
    y_test_prob1 = model.predict(test_padded)[:,1] 
    y_test_predict =  np.argmax(model.predict(test_padded), axis=-1) 

    test_f1 =np.round( f1_score(test_labels[:,1] , y_test_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Score: " + str(test_f1) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Score: " + str(test_f1) +"\n")
    
    test_recall = np.round(recall_score(test_labels[:,1], y_test_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Recall: " + str(test_recall) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Recall: " + str(test_recall) +"\n")
    
    test_precision =np.round(precision_score(test_labels[:,1], y_test_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Precision: " + str(test_precision) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Precision: " + str(test_precision) +"\n")
    
    test_accuracy =np.round( accuracy_score(test_labels[:,1], y_test_predict),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Accuracy: " + str(test_accuracy) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Accuracy: " + str(test_accuracy) +"\n")
    
    test_roc =np.round( roc_auc_score(test_labels[:,1], y_test_prob1),3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test ROC: " + str(test_roc) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test ROC: " + str(test_roc) +"\n")
    
    test_lift =np.round( (precision_score(test_labels[:,1], y_test_predict)/ (test_labels[:,1].sum()/len(test_labels[:,1]))) ,3) 
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Lift: " + str(test_lift) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline "+ "Test Lift: " + str(test_lift) +"\n")
    
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_labels[:,1], y_test_predict).ravel()
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline Test "+ " tn: " + str(tn_test) + " fp: " + str(fp_test) + " fn: " + str(fn_test) + " tp: " + str(tp_test) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline Test "+ " tn: " + str(tn_test) + " fp: " + str(fp_test) + " fn: " + str(fn_test) + " tp: " + str(tp_test) +"\n")
    
    f.write(str(dt.datetime.now())+"___"+"Step17_Best Pipeline Test "+ "positive ratio of test: "+ str(((fn_test+tp_test)/(tn_test+fp_test+fn_test+tp_test ))) +"\n")
    print(str(dt.datetime.now())+"___"+"Step17_Best Pipeline Test "+ "positive ratio of test: "+ str(((fn_test+tp_test)/(tn_test+fp_test+fn_test+tp_test ))) +"\n")

    
    #%% Create tsc files for tensorflow embedding projector
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_sentence(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])  
    
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape) # shape: (vocab_size, embedding_dim)
    
    
    import io
    
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, len(word_index)):
      word = reverse_word_index[word_num]
      embeddings = weights[word_num]
      out_m.write(word + "\n")
      out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()


#%%
    #-----------------------------------------------------------------------------#        
    f.close()
    sys.exit(0)
    #-----------------------------------------------------------------------------#
except Exception as e:
#    f.write("Python Script Stopped Error: "+str(e)+" Error Line {}".format(sys.exc_info()[-1].tb_lineno)+' At Generalized_Local_Propensity_Training.py'+'\n')
    print("Python Script Stopped Error: "+str(e)+" Error Line {}".format(sys.exc_info()[-1].tb_lineno)+' At Generalized_Local_Propensity_Training.py'+'\n')
#    f.close()
    sys.exit(-1)
    