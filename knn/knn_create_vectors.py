import json
import csv
import sys
import os
import math
import collections
from collections import defaultdict
import re
def getCmdArg():
    '''
    Function: This function get the user second input comand line argument. It check whether the input 
    number of command line arguments are correct, whether the file names input order are correct.
    Argument: None
    Return: a tsv file path, and a json path
    '''
    try:
        numCommandArg = 3           
        if len(sys.argv) != numCommandArg:
            sys.stderr.write('Give more or less command line arguments, make sure the amount of arguments is correct\n')
            sys.exit()

        if not((sys.argv[2].endswith('.tsv')) and (sys.argv[1].endswith('.json'))):
            sys.stderr.write('tsv file or json file not in correct order or name not correct\n')
            sys.exit()

        knnTsvPath = str(sys.argv[2])
        trainJsonPath = str(sys.argv[1])

    except Exception:
        sys.stderr.write('User input argument error, not found file/folder name\n' +
                         'One of the file names input incorrectly\n')
        sys.exit()
    return knnTsvPath, trainJsonPath

def check_tsv_file_in_path(file_path):
    '''
    This function serves for training program write to tsv that whether or not overwrite a file,
    It checks if the folder contains the file name, return True, else false, for later 
    write to tsv file function use.

    Argument: file path user input, that is sys.argv[2]

    return: boolean value
    '''
    try:
      path_parts = file_path.split("/")
      path_to_file = "/".join(path_parts[:-1])
      tsv_file_name = path_parts[-1]
      for file_name in os.listdir(path_to_file):
          if file_name.endswith(".tsv") and file_name == tsv_file_name:
              return True
    except Exception:
      sys.stderr.write('Error ---> No such file | directory found, please give a valid directory again\n')
      sys.exit()

def checkFormat(data):
  ''' 
  Function: 
  Arguments:
    data: json object
  Return:
    Boolean: return true if all the checks pass, otherwise, print to stderr and exit the program
  '''
  for doc in data:
    keys = doc.keys()
    if len(keys) != 2:
      sys.stderr.write("Error format in jsonFile\n")
      sys.exit()
    if "category" not in keys:
      sys.stderr.write("Error format in jsonFile, lack of category key")
      sys.exit()
    if "text" not in keys:
      sys.stderr.write("Error format in jsonFile, lack of text key")
      sys.exit()
      #################### Q: what if text is empty? or category is emtpy?
        # if doc[key] == "":
          # sys.stderr.write("no text")
          # sys.exit()
  return True

def getData(path):
  '''
  Function:
    Load json data from a json file.
  Arguments:
    path: a string of path pointing to the json file
  Return:
    data: a json object
  '''
  try:

    with open(path, "r") as f:
      data = json.load(f)
  except Exception as e:
    
    sys.stderr.write("inappropriate arguments, try again\n")
    sys.exit()
  return data

def getCandD(trainingData):
  '''
  Function:
      Get a list of classes names, a list of documents, and a dictionary storing docs for each class
  Arguments:
      trainingData: list type, the data loaded from jsonfile
  Return:
      classes: a list of no-repeated class names
      documents: a list of all the documents(text) from jsonfile
      classDict: a dictionary, with key=<class_name>, value=<a list of documents that belongs to the class>
  '''
  classes = list(set([i['category'] for i in trainingData]))
  documents = [i['text'] for i in trainingData]
  classDict = defaultdict(list)
  for i in trainingData:
    classDict[i['category']].append(i['text'])
  return classes, documents, classDict

def tokenize_and_filter(input_string):
  '''
  The tokenize_and_filter function takes an input string as its parameter and returns a list of filtered tokens. 
  This function performs three main operations on the input string. Firstly, it removes unwanted characters from the string using regular expressions. 
  Secondly, it tokenizes the string by splitting it into individual words and punctuation marks. Lastly, it filters out stopwords from the list of tokens. 
  Stopwords are common words that do not carry much meaning, such as articles and prepositions. The filtered tokens are then returned as the output of the function.

  Parameter: input_string which is the input text that needs to be processed. The input text can be of any length and can contain any combination of characters. The function is designed to handle all types of input strings.

  Return: a list of filtered tokens. 
  '''
  # Remove unwanted characters
  input_string = re.sub(r'&#', ' ', input_string)
  input_string = input_string.replace('\\u00a3', '£')
  
  # Tokenize the string
  tokens = re.findall(r"[\w'-]+|[.,!?;$£]", input_string)
  
  # Define stopwords
  stopwords = {'few', 'off', 'doing', 'over', 'in', 'how', 'some', 'own', "aren't", 'most', 'ours', 'am', 
  'the', 'hers', 'been', 'were', 'very', 'haven', 'theirs', 'has', 'have', 'him', 'whom', 'yourselves', 
  "haven't", 'ma', 'an', 'aren', 'now', 'a', 'more', 'so', 'here', 'and', 'during', 'further', 've', 'his', 
  'same', 'wouldn', 'through', 'we', 'shouldn', 'hasn', 'itself', 'won', "don't", 'i', "shan't", 'until', 'wasn', 
  'about', 'hadn', 'nor', "won't", "couldn't", "mightn't", "isn't", 'can', 'they', 'be', 'down', 'up', 'ain', 'couldn', 
  'should', 'mightn', 'because', 'themselves', 'by', 'these', 'mustn', 'both', 'd', "it's", 'she', 'do', 'yourself', 
  'while', 't', 'them', "weren't", "hasn't", 'if', 'under', 'against', "you'd", 'those', 'why', 'had', 'at', 'me', 'out', 
  'only', 'm', 'then', 'but', 'herself', 'isn', "mustn't", 'after', 's', 'myself', 'too', 'other', 'there', 'where', 'once', 
  'you', 'himself', 'to', 'all', 'not', 'below', 'their', 're', 'being', "you're", 'needn', 'yours', 'he', 'her', 'y', 
  'before', 'with', 'of', 'having', 'for', 'when', 'who', 'any', 'as', "you'll", 'ourselves', "she's", 'than', 'each', 
  'its', 'on', 'no', 'weren', "you've", 'between', "needn't", 'o', 'just', 'into', "should've", 'does', 'from', 'didn', 
  'it', 'doesn', 'your', "wasn't", 'or', 'such', "doesn't", 'this', 'll', "hadn't", "that'll", "didn't", 'is', 'are', '-are'
  "wouldn't", 'above', 'will', 'that', 'again', 'shan', 'what', 'which', "shouldn't", 'my', 'our', 'did', 'was', 'don','.',',','?'}
  
  # Filter out stopwords
  filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
  
  return filtered_tokens

def getTermFrequency(doc):
  '''
  Function: Calculates the term frequency of each word in a given document.
  
  Parameters: doc - A string representing the document whose term frequency is to be calculated.
  
  Returns: A dictionary containing the term frequency of each word in the document.
  '''
  words = tokenize_and_filter(doc)
  word_counts = collections.Counter(words)
  tf_dict = dict(word_counts)
  return tf_dict

def calcDocFrequency(D):
  '''
  Functionality: Calculates the document frequency of each term in a given list of documents.
  
  Parameters: D - A list of strings representing the documents whose document frequency is to be calculated.
  
  Returns: 
  A tuple containing two elements -
    A dictionary containing the document frequency of each term in the given list of documents.
    An integer representing the number of documents in the list.
  '''
  term_list = []
  for doc in D:
      terms = tokenize_and_filter(doc)
      terms = list(set(terms))
      terms.sort()
      for term in terms:
          if term == "":
              continue
          temp_dict = {}
          temp_dict[term] = 1
          term_list.append(temp_dict)
  merged_dict = {}
  for dic in term_list:
      for k,v in dic.items():
          if k not in merged_dict:
              merged_dict[k] = 1
          else:
              merged_dict[k] += 1
  df_dict = {k: v for k, v in sorted(merged_dict.items(), key=lambda item: item[0])}
  return df_dict, len(D)

def calcIdf(df_dict, N):
  '''
  Functionality: Calculates the inverse document frequency of each term using the document frequency 
  dictionary and the total number of documents.
  
  Parameters:
  df_dict - A dictionary containing the document frequency of each term.
  N - An integer representing the total number of documents in the list.
  
  Returns: A dictionary containing the inverse document frequency of each term.
  '''
  idfs = {}
  for term, df in df_dict.items():
      idfs[term] = math.log(N / df)
  idfs = dict(sorted(idfs.items()))
  return idfs

def calcWeights(tf_dict,idf_dict):
  '''
  Function: Calculates the weight of each term in a document based on the term frequency (tf) and inverse document frequency (idf).
  
  Parameters:
  tf_dict: A dictionary containing the term frequency of each term in the document.
  idf_dict: A dictionary containing the idf value of each term in the document.
  
  Returns: A dictionary containing the weight of each term in the document.
  '''
  weight_dict = {}
  norm = 0
  for term in tf_dict.keys():
      tf = tf_dict[term]
      logarithm_tf = 1 + math.log(tf)
      t_idf = idf_dict[term]
      weight = logarithm_tf * t_idf
      weight_dict[term] = weight
      norm += weight ** 2
  norm = math.sqrt(norm)
  normalized_weight_dict = {key: value / norm for key, value in weight_dict.items()}
  return normalized_weight_dict

def calcVectors(C,dic,idf):
  '''
  Function: Calculates the weighted vector representation of each document in the corpus.
  
  Parameters:
  C: A list of categories.
  dic: A dictionary containing the documents in each category.
  idf: A dictionary containing the idf value of each term in the corpus.
  
  Returns: A list of vectors, where each vector is a list containing the category and the weighted vector representation of a document.
  '''
  vectors = []
  for c in C:
    docs = dic[c]
    for doc in docs:
      tf_dict = getTermFrequency(doc)
      vector = calcWeights(tf_dict, idf)
      vectors.append([c,vector])
  return vectors

def writeToTSV(path, idfs, vectors):
  '''
  Functionality: Writes the idf values and the vector representations of the documents to a TSV file.
  
  Parameters:
  path: The path to the TSV file to be written.
  idfs: A dictionary containing the idf value of each term in the corpus.
  vectors: A list of vectors, where each vector is a list containing the category and the weighted vector representation of a document.
  
  Returns: None. The function only writes the data to the specified TSV file.
  '''
  with open(path, 'wt', encoding='utf-8') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for vector in vectors:
      tsv_writer.writerow(["vector",vector[0],vector[1]])
    for term,idf in idfs.items():
      tsv_writer.writerow(["idf",term,idf])

def main():
  knnTsvPath, trainJsonPath = getCmdArg()
  isFileExists = check_tsv_file_in_path(knnTsvPath)
  if isFileExists:
    while True:
      choice =input(f"Are you sure you want to overwrite {knnTsvPath}? (y/n): ")
      if choice.lower() == "y":
        break
      elif choice.lower() =='n':
        return
      else:
        print("Invalid input, please enter 'y' or 'n'")
  data = getData(trainJsonPath)
  checkFormat(data)
  C,D,dic = getCandD(data)
  df, N = calcDocFrequency(D)
  idf = calcIdf(df,N)
  vectors = calcVectors(C,dic,idf)
  writeToTSV(knnTsvPath, idf, vectors)
    

if __name__ == "__main__":
    main()