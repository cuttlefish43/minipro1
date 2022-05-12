from cmath import e
from django.apps import AppConfig
import pickle
from transformers import PegasusForConditionalGeneration, PegasusTokenizer,pipeline
import os
""" global tokenizer_model
global loaded_model """
class SummarizeConfig(AppConfig):
    name = 'summarize'
    print("model initializing")
    tokenizer_model={}
    loaded_model={}
    sentiment_model={}
    try:
        tokenizer_model=pickle.load(open("tokenizer_model.pkl","rb")) 
        print("tokenizer model unpickled")   
    except OSError:
        tokenizer_model = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        pickle.dump(tokenizer_model,open("tokenizer_model.pkl","wb"))
        print("tokenizer model redownloaded")   
    try:
        loaded_model=pickle.load(open("loaded_model.pkl","rb"))
        print("loaded model unpickled") 
    except OSError:
        loaded_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        pickle.dump(loaded_model,open("loaded_model.pkl","wb"))
        print("loaded model redownloaded")
    
    try:
        sentiment_model=pickle.load(open("sentiment_model.pkl","rb"))
        print("sentiment model unpickled") 
    except OSError:
        sentiment_model=pipeline("sentiment-analysis")
        pickle.dump(sentiment_model,open("sentiment_model.pkl","wb"))
        print("sentiment model redownloaded")
    print("models initialized")
