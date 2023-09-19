from bs4 import BeautifulSoup
import string
import os
import time
import gc
import pickle
from bs4.dammit import html_meta
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
from math import cos, log10 as log
from math import sqrt

t0 = time.time()

def filter_html_file_lemmatizer_with_stopword(soup_html_file):
    lemmatizer = WordNetLemmatizer()
    #set the stopwords to english
    stop_words_to_find = set(stopwords.words('english'))
    soup_html_file.prettify()
    #extract the useless links from the text such as '[citation needed]' and the reference links [1], [2] etc
    [x.extract() for x in soup_html_file.select(".noprint")]
    [x.extract() for x in soup_html_file.select(".reference")]

    #find all the key information about the character
    relevant_data_selector = soup_html_file.select("p")

    data_result_list = []
    #now we are going to filter each tag by removing stop words, punctuation etc
    for tag in relevant_data_selector: 
        translation_table = tag.text.maketrans('','',string.punctuation)
        translate_to_string = tag.text.translate(translation_table)
        tokenized_word = word_tokenize(translate_to_string)
        #append filtered, tokenized sentence to the data result list
        for word in tokenized_word:
            #lemmatize the word to simplify it
            data_result_list.append(lemmatizer.lemmatize(word).lower())

    return data_result_list

def filter_html_file_lemmatizer_without_stopword(soup_html_file):
    lemmatizer = WordNetLemmatizer()
    #set the stopwords to english
    stop_words_to_find = set(stopwords.words('english'))
    soup_html_file.prettify()
    #extract the useless links from the text such as '[citation needed]' and the reference links [1], [2] etc
    [x.extract() for x in soup_html_file.select(".noprint")]
    [x.extract() for x in soup_html_file.select(".reference")]

    #find all the key information about the character
    relevant_data_selector = soup_html_file.select("p")

    data_result_list = []
    #now we are going to filter each tag by removing stop words, punctuation etc
    for tag in relevant_data_selector: 
        translation_table = tag.text.maketrans('','',string.punctuation)
        translate_to_string = tag.text.translate(translation_table)
        tokenized_word = word_tokenize(translate_to_string)
        filtered_sentence = [w for w in tokenized_word if not w in stop_words_to_find]
        #append filtered, tokenized sentence to the data result list
        for word in filtered_sentence:
            #lemmatize the word to simplify it
            data_result_list.append(lemmatizer.lemmatize(word).lower())

    return data_result_list

def filter_html_file_stemmer_without_stopword(soup_html_file):
    porter_stemmer = PorterStemmer()
    #set the stopwords to english
    stop_words_to_find = set(stopwords.words('english'))
    soup_html_file.prettify()
    #extract the useless links from the text such as '[citation needed]' and the reference links [1], [2] etc
    [x.extract() for x in soup_html_file.select(".noprint")]
    [x.extract() for x in soup_html_file.select(".reference")]

    #find all the key information about the character
    relevant_data_selector = soup_html_file.select("p")

    data_result_list = []
    #now we are going to filter each tag by removing stop words, punctuation etc
    for tag in relevant_data_selector: 
        translation_table = tag.text.maketrans('','',string.punctuation)
        translate_to_string = tag.text.translate(translation_table)
        tokenized_word = word_tokenize(translate_to_string)
        filtered_sentence = [w for w in tokenized_word if not w in stop_words_to_find]
        #append filtered, tokenized sentence to the data result list
        for word in filtered_sentence:
            #lemmatize the word to simplify it
            data_result_list.append(porter_stemmer.stem(word).lower())

    return data_result_list

def filter_html_file_stemmer_with_stopword(soup_html_file):
    porter_stemmer = PorterStemmer()

    soup_html_file.prettify()
    #extract the useless links from the text such as '[citation needed]' and the reference links [1], [2] etc
    [x.extract() for x in soup_html_file.select(".noprint")]
    [x.extract() for x in soup_html_file.select(".reference")]

    #find all the key information about the character
    relevant_data_selector = soup_html_file.select("p")

    data_result_list = []
    #now we are going to filter each tag by removing stop words, punctuation etc
    for tag in relevant_data_selector: 
        translation_table = tag.text.maketrans('','',string.punctuation)
        translate_to_string = tag.text.translate(translation_table)
        tokenized_word = word_tokenize(translate_to_string)
        #append filtered, tokenized sentence to the data result list
        for word in tokenized_word:
            #lemmatize the word to simplify it
            data_result_list.append(porter_stemmer.stem(word).lower())

    return data_result_list

rootdir = os.getcwd()
fileTitle = []
f = []

def save_data():
    for subdir,dirs,files in os.walk(rootdir+"\\ueapeople"):
        for file in files:
            file_path = subdir+os.sep + file
            if file_path.endswith(".html"):
                print(file_path)
                soup = BeautifulSoup(open(file_path,"rb"),'lxml')
                html_file_name = str(soup.title.text)
             
                f.append(filter_html_file_stemmer_without_stopword(soup))
                fileTitle.append(html_file_name[:33] if len(html_file_name) > 20 else html_file_name )
            
    
    with open("data.txt","wb") as fp:
        pickle.dump(f, fp)
    with open("file_title.txt","wb") as fp:
        pickle.dump(fileTitle,fp)





#save_data()


tokens = []


with open("file_title.txt","rb") as fp:
    fileTitle = pickle.load(fp)

with open("data.txt","rb") as fp:
    tokens = pickle.load(fp)


flattened_tokens = chain(*tokens)
token_counts = Counter(flattened_tokens)

unique_token_list = [term for term in token_counts]

del flattened_tokens
gc.collect()
del token_counts
gc.collect()

query = "Give me a course that will benefit my programming skills in general"
query_tok = word_tokenize(query.lower())
ps = PorterStemmer()

b = [ps.stem(w) for w in query_tok]
print(b)
c = Counter(query_tok)

incidence_matrix  = {tokenTerm:[int(tokenTerm in token) for token in tokens] for tokenTerm in unique_token_list}

print("matrix created")



def create_term_freq_vec_query(unique_token_query_list,counter_q):
    term_freq_vec = []
    for unique_token in unique_token_list:
        term_freq_vec.append(unique_token_query_list.count(unique_token))

    return term_freq_vec

def create_term_freq_vec(unique_token_list):
    term_frequency_vector = ((token.count(unique_token) for unique_token in unique_token_list) for token in tokens)
    
    return term_frequency_vector

term_frequency_vector = create_term_freq_vec(unique_token_list)


print("term frequency done")

document_frequency = (incidence_matrix[token].count(1) for term in unique_token_list for token,val in incidence_matrix.items() if term == token)


print("doc frequency done")


def create_def_weight_query(freq_vector):
    tf_weight = (log(1+term_freq) if term_freq > 0 else 0 for term_freq in freq_vector)
    return tf_weight

def create_def_weight(freq_vector):
    tf_weight = ((log(1+freq) if freq > 0 else 0 for freq in term_freq ) for term_freq in freq_vector)
    return tf_weight

def create_def_weight_query_boolean(freq_vector):
    tf_weight = (1 if term_freq > 0 else 0 for term_freq in freq_vector)
    return tf_weight

def create_def_weight_boolean(freq_vector):
    tf_weight = ((1 if freq > 0 else 0 for freq in term_freq ) for term_freq in freq_vector)
    return tf_weight

tf_weight = ((1 if freq > 0 else 0 for freq in term_freq ) for term_freq in term_frequency_vector)

print("tf weight done")
#amount of documents
n = len(tokens)



idf = [log(n/freq)  if freq > 0 else 0 for freq in document_frequency]

del incidence_matrix
gc.collect()

idf_q = (val for val in idf)



def create_tf_idf_query(tf_weight_q,idf_weight_q):
    tf_idf = []
    for w,i in zip(tf_weight_q,idf_weight_q):
        tf_idf.append(w*i)
    return tf_idf

print("idf weight done")

q_frequency_vector = create_term_freq_vec_query(query_tok,c)


q_freq_weight = create_def_weight_query_boolean(q_frequency_vector)

q_tf_idf = create_tf_idf_query(q_freq_weight,idf_q)




tf_idf = ((freq_weight*idf_weight for freq_weight,idf_weight in zip(tf,idf)) for tf in tf_weight)




print("tf idf done")


def vec_length(vec):
    sq_vec = (x**2 for x in vec)
    vec_sum = sum((vec for vec in sq_vec))
    return sqrt(vec_sum)


def vec_normalisation(vec):
    magnitude = vec_length(vec)
    vec_normalised = (value/magnitude if value > 0 else 0 for value in vec)
    return vec_normalised


def dot_product(vec1,vec2):
    product = sum((v1*v2 for v1,v2 in zip(vec1,vec2)))
    return product


def cosine_scores():

    result_list = {}
    for index, value in enumerate(tf_idf):
        query_norm = vec_normalisation(q_tf_idf)
        normalised_doc = vec_normalisation(list(value))
        result_list[fileTitle[index]] = dot_product(query_norm,normalised_doc)
        score_ranked = sorted(result_list.items(), key=lambda x: x[1], reverse=True)[:10]
        dict_score = dict((doc_name, weight) for doc_name, weight in score_ranked)
    return dict_score


cosine_score_list = cosine_scores()

del unique_token_list
gc.collect()

for score,value in cosine_score_list.items():
    print(score,":",value)




def show_top_ten_doc_barchart(rank):
    x = [key for key, value in rank.items()]
    y = [value for key, value in rank.items()]


  
    fig = plt.figure(figsize=(10, 6))

    plt.bar(x, y, width=0.3)
    plt.xlabel("documents", fontsize = 15)
    plt.ylabel("weighting", fontsize = 15)
    plt.title("top 10 documents")

    plt.setp(fig.get_axes()[0].get_xticklabels(), rotation=40)
    plt.xticks(fontsize = 8)
    plt.tight_layout()
    plt.savefig('output.png')
    plt.show()

t1 = time.time()
full_time = t1-t0
print("TIME IT TOOK - ",full_time)


show_top_ten_doc_barchart(cosine_score_list)






