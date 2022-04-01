# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
import re
import nltk
import string
import numpy as np
import networkx as nx
from nltk.cluster.util import cosine_distance
# Create your views here.
""" def home(request):
    return HttpResponse("Hello world") """
############################################################ Algo 1
nltk.download('punkt')
    #nltk.download('stopwords')
    #stopwords = nltk.corpus.stopwords.words('english')
stopwords=['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
def preprocess(text):
    formatted_text = text.lower()
    tokens = []
    for token in nltk.word_tokenize(formatted_text):
        tokens.append(token)
    tokens = [word for word in tokens if word not in stopwords and word not in string.punctuation]
    formatted_text = ' '.join(element for element in tokens)

    return formatted_text
def calculate_sentence_similarity(sentence1, sentence2):
    words1 = [word for word in nltk.word_tokenize(sentence1)]
    words2 = [word for word in nltk.word_tokenize(sentence2)]
    #print(words1)
    #print(words2)

    all_words = list(set(words1 + words2))
    #print(all_words)

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    #print(vector1)
    #print(vector2)

    for word in words1: # Bag of words
        #print(word)
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1
    
    #print(vector1)
    #print(vector2)

    return 1 - cosine_distance(vector1, vector2)
def calculate_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = calculate_sentence_similarity(sentences[i], sentences[j])
    return similarity_matrix
#calculate_similarity_matrix(formatted_sentences)
def summarize(text, number_of_sentences, percentage = 0):
    original_sentences = [sentence for sentence in nltk.sent_tokenize(text)]
    formatted_sentences = [preprocess(original_sentence) for original_sentence in original_sentences]
    similarity_matrix = calculate_similarity_matrix(formatted_sentences)
    #print(similarity_matrix)

    similarity_graph = nx.from_numpy_array(similarity_matrix)
    #print(similarity_graph.nodes)
    #print(similarity_graph.edges)

    scores = nx.pagerank(similarity_graph)
    #print(scores)
    ordered_scores = sorted(((scores[i], score) for i, score in enumerate(original_sentences)), reverse=True)
    #print(ordered_scores)

    if percentage > 0:
        number_of_sentences = int(len(formatted_sentences) * percentage)

    best_sentences = []
    for sentence in range(number_of_sentences):
        best_sentences.append(ordered_scores[sentence][1])
    great_sentences=""
    for sentence in range(number_of_sentences):
        great_sentences+=ordered_scores[sentence][1]
    
    return great_sentences, ordered_scores
input_txt=" "
result,score=summarize(input_txt, 120,0.2)
    #print(result)
############################################################
def home(request):
    if request.method == 'POST':
        #text_input=request.inptext
        #do processing on this txt
        inp_text=request.POST['inptext']
        summarized_output,score=summarize(inp_text,120,0.2)
        # print(summarized_output)
        return render(request,'home.html',{'input_txt':inp_text,'output_txt':summarized_output})

    else:
        input_txt="Please paste input your text here"
        output_txt="No input to generate output"
        return render(request,'home.html',{'input_txt':input_txt,'output_txt':output_txt})


    
