import re
# from porter import PorterStemmer
import os
import math
from nltk.stem import WordNetLemmatizer
import json
import numpy
import pathlib

#StopWord File reading
def preprocessing(stopwords,tf_idf_index):
    try:
        # Load tf_idf if its already saved
        with open("data_file.json", "r") as write_file:
            json.load(tf_idf_index, write_file)
        return tf_idf_index
    except:
        path = pathlib.Path(__file__).parent.absolute()   
        lemmatizer = WordNetLemmatizer()  
        path = str(path)
        stopwords_file=open(path + "\\Stopword-List.txt", "r", encoding = "utf-8")
        Lines=stopwords_file.readlines()
        for line in Lines:
            stopwords.append(line.strip()) 
        stopwords=[i for i in stopwords if i]
        stopwords_file.close()

        #DataSet File reading
        temp_dict={}
        dataset=[]
        temp2=''
        i=0
        TC=1    

        for docid in range(1,51):
            #Reset index for new file
            index=0
            #Read next file
            path = pathlib.Path(__file__).parent.absolute()     
            path = str(path)
            text=open(path+"\\ShortStories\\"+str(docid)+".txt","r",encoding="utf-8")
            temp=text.read()
            temp=temp.lower()
            temp2=temp
            #Regex Matching to remove special characters
            temp=re.sub(r"[-—“”’()\"#:<>{}*`+=~|.!/@;?,]", "", temp)
            temp=temp.split()       
            temp=[word for word in temp if not word in stopwords]
            temp=[ x for x in temp if x]
            #TermFrequency-idf index
            for word in temp:
                #if exist
                word=lemmatizer.lemmatize(word)
                if word in tf_idf_index.keys():
                    #If word appeared in same doc or not
                    if docid in tf_idf_index[word][1]:
                        #Increment term frequency
                        tf_idf_index[word][1][docid]=tf_idf_index[word][1][docid]+1
                    #Append list with 1st index having total word length in doc and the other tracking word frequency
                    else:
                        tf_idf_index[word][1][docid]=1

                #if not
                else:
                    #Index for IDF
                    tf_idf_index[word]=[]
                    #Initialize idf field
                    tf_idf_index[word].append(0)
                    tf_idf_index[word].append({})
                    tf_idf_index[word][1][docid]=1
                index+=1
            
            text.close()
            for word in tf_idf_index:
                # print(word)
                # print(len(tf_idf_index[word][1]))
                if word=='hospital':
                    print
                tf_idf_index[word][0]=math.log10(len(tf_idf_index[word][1]))/50
        #Save the tf_idf index
        with open("data_file.json", "w") as write_file:
            json.dump(tf_idf_index, write_file)
        return tf_idf_index
        

def query_processing(query):
    stopwords=[]
    path = pathlib.Path(__file__).parent.absolute()   
    lemmatizer = WordNetLemmatizer()  
    path = str(path)
    stopwords_file=open(path + "\\Stopword-List.txt", "r", encoding = "utf-8")
    Lines=stopwords_file.readlines()
    for line in Lines:
        stopwords.append(line.strip()) 
    stopwords=[i for i in stopwords if i]
    stopwords_file.close()
    # print(query)
    alpha=0.05
    with open('data_file.json') as f:
        tf_idf_index=json.load(f)
    print(len(tf_idf_index))
    with open('doc_matrix.json') as f:
        doc_matrix=json.load(f)
    print(len(tf_idf_index))
    # return 0
    query=re.sub(r"[-—“”’()\"#:<>{}*`+=~|.!/@;?,]", "", query)
    query=query.lower().split()
    for q in query:
        q=lemmatizer.lemmatize(q)
    query=[word for word in query if not word in stopwords]
    print(query)
    words=[]
    query_vector=[]
    for word in tf_idf_index.keys():
        if word in query:
            query_vector.append(tf_idf_index[word][0])
        else:
            query_vector.append(0)
    query_magnitude=math.sqrt(sum(q**2 for q in query_vector))
    cosine_similarity={}
    for doc in range(1,51):
        data=doc_matrix[str(doc)]
        doc_magnitude=math.sqrt(sum(d**2 for d in data))
        ans=numpy.dot(data,query_vector)
        if doc_magnitude!=0:
            ans=ans/(doc_magnitude*query_magnitude)
        elif query_magnitude!=0:
            ans=ans/(doc_magnitude*query_magnitude)
        #Adjust Alpha value here
        if ans>=alpha:
            cosine_similarity[doc]=ans
    return cosine_similarity

def document_matrix_creation():
    with open('data_file.json') as f:
        tf_idf_index=json.load(f)
    document_matrix={}
    words=list(tf_idf_index.keys())
    #initialize doc matrix
    for d in range(1,51):
        document_matrix[d]=[0]*len(words)
    #Filling document matrix
    index=0
    for word in words:
        data=tf_idf_index[word][1]
        for key in data.keys():
            #Fill Docmatrix doc id with Index of the current word  * by IDF in termfreq index at 0th
            document_matrix[int(key)][index]=data[key]*tf_idf_index[word][0]
        index=index+1
    with open("doc_matrix.json", "w") as write_file:
        json.dump(document_matrix, write_file)

        

# stopwords=[]
# inverted_index = {}
# tf_idf_index= {}
# preprocessing(stopwords,tf_idf_index)
# document_matrix_creation()
# query='of agony'
# ans=query_processing(query)
# # print(inverted_index)
# print(query)
# for a in ans:
#     print(a)
# print(len(ans),'total')
