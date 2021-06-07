import spacy
import scipy
import math
from scipy import spatial
from spacy.lang.en import English
from collections import Counter
from string import punctuation
from spacy.vectors import Vectors
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from scipy.spatial import distance
import matplotlib.pyplot as plt
import networkx as nx

# Build a List of Stopwords
stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_md')
file_name ='D:\Python\miniproject\dataset1.story'
introduction_file_text = open(file_name, encoding="utf8").read()
introduction_file_doc = nlp(introduction_file_text)

# Extract tokens for the given doc
sentences = list(introduction_file_doc.sents)
numberOfSentences=len(sentences)
sent_vectors = list()
################################################################################

for sentence in sentences:
    sentence=(sentence.text).lower()
    sentence=nlp(sentence)
    count=0
    sum=[]
    for x in range(0,300):
        sum.append(0)
    for word in sentence:
         if word.is_stop==False and word.is_punct==False:
            count+=1
            new_word=nlp(word.lemma_)
            word_vector=new_word.vector
            for i in range(len(word_vector)):
                sum[i] = sum[i] + word_vector[i]
    if count!=0:
        sum[:] = [x / count for x in sum]
    sent_vectors.append(sum)

######################################################################################

similarity_matrix=np.zeros((numberOfSentences,numberOfSentences))
for row in range (numberOfSentences):
    for column in range (numberOfSentences):
        a=sent_vectors[row]
        b=sent_vectors[column]
        similarity_matrix[row][column]= 1 - spatial.distance.cosine(a,b)


#######################################################################################
adjacency_matrix=np.zeros((numberOfSentences,numberOfSentences))
threshold = 0.7
for row in range (numberOfSentences):
    for column in range (numberOfSentences):
      if(row==column):
        adjacency_matrix[row][column]=0
      elif(similarity_matrix[row][column]>threshold):
        adjacency_matrix[row][column]=1


#######################################################################################
#PAGE RANK ALGO
def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
         v = M_hat @ v
    return v

v = pagerank(adjacency_matrix, 100, 0.85)
#print(v)



v_arr = []

for x in v:
 v_arr.append(x[0])

print(v_arr)

##########################################
sort_index = np.argsort(v_arr)
print(sort_index)

sort_index=np.flip(sort_index)
print(sort_index)

store_index = []
num = 0.2 * len(sentences)
num = math.floor(num)

for i in range(num):
    store_index.append(sort_index[i])

store_index.sort()

summary = []

for i in range(num):
 summary.append(sentences[store_index[i]])

print(*summary, sep = " ")

print(len(summary))
print(len(sentences))


G = nx.DiGraph()

for row in range (numberOfSentences):
 for column in range (numberOfSentences):
  if adjacency_matrix[row][column]==1:
    G.add_edge(row,column)

nx.draw( G )
plt.show()
