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
import heapq
from nltk.cluster.util import cosine_distance
# Create your views here.
""" def home(request):
    return HttpResponse("Hello world") """

############################################################ Algo 1
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
#stopwords=['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
class A:
    
    def __init__(self,text):
        self.text=text
    def preprocess(self,text):
        formatted_text = text.lower()
        tokens = []
        for token in nltk.word_tokenize(formatted_text):
            tokens.append(token)
        tokens = [word for word in tokens if word not in stopwords and word not in string.punctuation]
        formatted_text = ' '.join(element for element in tokens)
    
        return formatted_text
    def calculate_sentence_similarity(self,sentence1, sentence2):
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
    def calculate_similarity_matrix(self,sentences):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                similarity_matrix[i][j] = self.calculate_sentence_similarity(sentences[i], sentences[j])
        return similarity_matrix
    #calculate_similarity_matrix(formatted_sentences)
    def summarize( self,number_of_sentences, percentage = 0):
        original_sentences = [sentence for sentence in nltk.sent_tokenize(self.text)]
        formatted_sentences = [self.preprocess(original_sentence) for original_sentence in original_sentences]
        similarity_matrix = self.calculate_similarity_matrix(formatted_sentences)
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
        for sentence in best_sentences:
            great_sentences+=sentence
        
        return great_sentences, ordered_scores

############################################################
class B:
    def __init__(self,text):
        self.text=text
        stopwords.append('explanation')
    def preprocess(self,text):
        formatted_text = text.lower()
        tokens = []
        for token in nltk.word_tokenize(formatted_text):
            tokens.append(token)
        tokens = [word for word in tokens if word not in stopwords and word not in string.punctuation]
        formatted_text = ' '.join(element for element in tokens)

        return formatted_text
    def calculate_sentences_score(self,sentences, important_words, distance):
        scores = []
        sentence_index = 0

        for sentence in [nltk.word_tokenize(sentence) for sentence in sentences]:

            word_index = []
            for word in important_words:
            #print(word)
                try:
                    word_index.append(sentence.index(word))
                except ValueError:
                    pass

            word_index.sort()
            #print(word_index)

            if len(word_index) == 0:
                continue

            # [0, 1, 5]
            groups_list = []
            group = [word_index[0]]
            i = 1 # 3
            while i < len(word_index): # 3
            # first execution: 1 - 0 = 1
            # second execution: 2 - 1 = 1
                if word_index[i] - word_index[i - 1] < distance:
                    group.append(word_index[i])
                #print('group', group)
                else:
                    groups_list.append(group[:])
                    group = [word_index[i]]
                #print('group', group)
                i += 1
            groups_list.append(group)
            #print('all groups', groups_list)

            max_group_score = 0
            for g in groups_list:
            #print(g)
                important_words_in_group = len(g)
                total_words_in_group = g[-1] - g[0] + 1
                score = 1.0 * important_words_in_group**2 / total_words_in_group
            #print('group score', score)

                if score > max_group_score:
                    max_group_score = score

            scores.append((max_group_score, sentence_index))
            sentence_index += 1

        #print('final scores', scores)
        return scores
    def summarize(self, top_n_words, distance, number_of_sentences, percentage = 0):
        stopwords.append('explanation')
        original_sentences = [sentence for sentence in nltk.sent_tokenize(self.text)]
        #print(original_sentences)
        formatted_sentences = [self.preprocess(original_sentence) for original_sentence in original_sentences]
        #print(formatted_sentences)
        words = [word for sentence in formatted_sentences for word in nltk.word_tokenize(sentence)]
        #print(words)
        frequency = nltk.FreqDist(words)
        #print(frequency)
        #return frequency
        top_n_words = [word[0] for word in frequency.most_common(top_n_words)]
        #print(top_n_words)
        sentences_score = self.calculate_sentences_score(formatted_sentences, top_n_words, distance)
        #print(sentences_score)
        if percentage > 0:
            best_sentences = heapq.nlargest(int(len(formatted_sentences) * percentage), sentences_score)
        else:  
            best_sentences = heapq.nlargest(number_of_sentences, sentences_score)
        #print(best_sentences)
        best_sentences = [original_sentences[i] for (score, i) in best_sentences]
        great_sentences=""
        for i in best_sentences:
            great_sentences+=i
        #print(best_sentences)
        return great_sentences, sentences_score


    #print(result)
############################################################
    
############################################################
def home(request):
    if request.method == 'POST':
        print(f"into post method")
        #text_input=request.inptext
        #do processing on this txt
        inp_text=request.POST['inptext']
        # print(f"inp_text {inp_text}")
        print(len(inp_text))
        if len(inp_text) <= 200:
            print("too short")
            input_txt=""
            output_txt="Input too short"
            return render(request,'home.html',{'input_txt':input_txt,'output_txt':output_txt})
        
        obj1=A(inp_text)
        text_output1,sc=obj1.summarize(120,0.3)
        
        print(f"text_output1 {text_output1}")

        #summarized_output,score=summarize(inp_text,120,0.3)
        obj2=B(inp_text)
        text_output2,sc=obj2.summarize(300,10,120,0.3)
        print(f"text_output2 {text_output1}")
        #print(text_output2)
        # print(summarized_output)
        return render(request,'home.html',{'input_txt':inp_text,'output_txt1':text_output1, 'output_txt2':text_output2})

    else:
        input_txt=""
        output_txt="No input to generate output"
        return render(request,'home.html',{'input_txt':input_txt,'output_txt':output_txt})


    
