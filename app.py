import streamlit as st
import pandas as pd
import os
import json
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag
import numpy as np
import pickle
import string
import random
import timeit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 


#greeting function
GREETING_INPUTS = ("hello", "hi", "greetings", "hello i need help", "good day","hey","i need help", "greetings")
GREETING_RESPONSES = ["Good day, How may i of help?", "Hello, How can i help?", "hello", "I am glad! You are talking to me."]
           
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


lemmer = nltk.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def RemovePunction(tokens):
    return[t for t in tokens if t not in string.punctuation]

stop_words = set(stopwords.words('english'))



def Talk_To_Applied_Ai_Chatbot(test_set_sentence):
    json_file_path = "conversation_json_ai.json" 
    tfidf_vectorizer_pickle_path = "tfidf_vectorizer_ai.pkl"
    tfidf_matrix_pickle_path = "tfidf_matrix_train_ai.pkl"
    
    i = 0
    sentences = []
    
    # ---------------Tokenisation of user input -----------------------------#
    
    tokens = RemovePunction(nltk.word_tokenize(test_set_sentence))
    pos_tokens = [word for word,pos in pos_tag(tokens, tagset='universal')]
    
    word_tokens = LemTokens(pos_tokens)
    
    filtered_sentence = []
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)  
    
    filtered_sentence =" ".join(filtered_sentence).lower()
            
    test_set = (filtered_sentence, "")
    
    #For Tracing, comment to remove from print.
    #print('USER INPUT:'+filtered_sentence)
    
    # -----------------------------------------------------------------------#
        
    try: 
        # ---------------Use Pre-Train Model------------------#
        f = open(tfidf_vectorizer_pickle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()
        
        f = open(tfidf_matrix_pickle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        # ---------------------------------------------------#
    except: 
        # ---------------To Train------------------#
        
        start = timeit.default_timer()
        
        with open(json_file_path) as sentences_file:
            reader = json.load(sentences_file)
            
            # ---------------Tokenisation of training input -----------------------------#    
            
            for row in reader:
                db_tokens = RemovePunction(nltk.word_tokenize(row['question']))
                pos_db_tokens = [word for word,pos in pos_tag(db_tokens, tagset='universal')]
                db_word_tokens = LemTokens(pos_db_tokens)
                
                db_filtered_sentence = [] 
                for dbw in db_word_tokens: 
                    if dbw not in stop_words: 
                        db_filtered_sentence.append(dbw)  
                
                db_filtered_sentence =" ".join(db_filtered_sentence).lower()
                
                #Debugging Checkpoint
                print('TRAINING INPUT: '+db_filtered_sentence)
                
                sentences.append(db_filtered_sentence)
                i +=1                
            # ---------------------------------------------------------------------------#
                
        tfidf_vectorizer = TfidfVectorizer() 
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)
        
        #train timing
        stop = timeit.default_timer()
        print ("Training Time : ")
        print (stop - start) 
    
        f = open(tfidf_vectorizer_pickle_path, 'wb')
        pickle.dump(tfidf_vectorizer, f) 
        f.close()
    
        f = open(tfidf_matrix_pickle_path, 'wb')
        pickle.dump(tfidf_matrix_train, f) 
        f.close 
        # ------------------------------------------#
        
    #use the learnt dimension space to run TF-IDF on the query
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)

    #then run cosine similarity between the 2 tf-idfs
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    
    #if not in the topic trained.no similarity 
    idx= cosine.argsort()[0][-2]
    flat =  cosine.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if (req_tfidf==0): #Threshold A
        
        not_understood = "I did not get that. Can you put your question differently?"
        
        return not_understood, not_understood, 2
        
    else:
        
        cosine = np.delete(cosine, 0)

        #get the max score
        max = cosine.max()
        response_index = 0

        #if max score is lower than < 0.34 > (we see can ask if need to rephrase.)
        if (max <= 0.34): #Threshold B
            
            not_understood = "I did not get that. Can you put your question differently?"
            
            return not_understood,not_understood, 2
        else:

                #if score is more than 0.91 list the multi response and get a random reply
                if (max > 0.91): #Threshold C
                    
                    new_max = max - 0.05 
                    # load them to a list
                    list = np.where(cosine > new_max) 
                   
                    # choose a random one to return to the user 
                    response_index = random.choice(list[0])
                else:
                    # else we would simply return the highest score
                    response_index = np.where(cosine == max)[0][0] + 2 

                j = 0 

                with open(json_file_path, "r") as sentences_file:
                    reader = json.load(sentences_file)
                    for row in reader:
                        j += 1 
                        if j == response_index: 
                            return row["answer"], row["question"], max
                            break



st.title("Applied-AI-Chatbot")
st.text("I try to answer all your questions related to the Applied-Ai-Course at Applied-courses")
st.text("you can exit any time using <exit>")


question = st.text_input('Ask your question here:')

if question:
    if(question.lower()!='exit'):
        if(greeting(question.lower())!=None):
            st.write(greeting(question.lower()))
            
        else:
            response_primary, response_message, line_id_primary = Talk_To_Applied_Ai_Chatbot(question)
            st.write(response_primary)
    else:
        st.write("Good day! Hope i answered some of yours questions") 