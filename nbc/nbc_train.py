import sys
import os
import json
import csv
import numpy as np

import re
from collections import defaultdict
from nltk.tokenize import word_tokenize


def getCmdArg():
    '''
    Function: This function get the user second input comand line argument. It check whether the input 
    number of command line arguments are correct, whether the file names input order are correct.
    Argument: None
    Return: a tsv file path, and a json path
    '''
    try:
        numCommandArg = 3           # For NBC prediction program, the number of command arguments is 3 = 1 python + 2 tsv and test json paths
        if len(sys.argv) != numCommandArg:
            sys.stderr.write('Give more or less command line arguments, make sure the amount of arguments is correct\n')
            sys.exit()

        if (sys.argv[2].endswith('.tsv')) and ('json' not in sys.argv[1]):
            sys.stderr.write('tsv file or json file not in correct order or name not correct\n')
            sys.exit()

        nbcTSVPath = str(sys.argv[2])
        trainJsonPath = str(sys.argv[1])

    except Exception:
        sys.stderr.write('User input argument error, not found file/folder name\n' +
                         'One of the file names input incorrectly\n')
        sys.exit()
    return nbcTSVPath, trainJsonPath

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
def extractVolcabulary(docs):
  '''
  Function:
    extract vocabularies from a list of strings
  Argument:
    docs: a list of strings
  Return:
    vocabularies: a list of vocabularies
  '''
  vocab = []
  for string in docs:
    token = tokenize_and_filter(string)
    vocab.extend(token)
  vocab = list(set(vocab))
  return vocab

def trainMultinomialNB(C, D, dic):
  '''
  Function:
  Trains a Multinomial Naive Bayes model using the given training data.
  Returns the model's vocabulary, prior probabilities for each class, and conditional probabilities for each term in the vocabulary.

  Parameters:
  C (list): List of classes in the training data.
  D (list): List of documents in the training data.
  dic (dict): Dictionary mapping each class to a list of documents belonging to that class.

  Returns:
  vocab (list): List of unique terms in the training data.
  priorList (list): List of prior probabilities for each class in the format [[class1, prior1], [class2, prior2], ...].
  condProbList (list): List of conditional probabilities for each term in the vocabulary in the format [[class1, term1, condProb1], [class2, term2, condProb2], ...].
  '''
  N = len(D)
  priorList = []
  condProbList = []
  
  # Calculate prior probability for each class
  docCounts = {c: len(docs) for c, docs in dic.items()}
  vocab = extractVolcabulary(D)
  for c in C:
    Nc = docCounts[c]
    
    prior = Nc / N
    priorList.append([c, format(prior,'.10f')])

    # Calculate conditional probability for each term in the vocabulary
    termCounts = defaultdict(int)
    for doc in dic[c]:
      tokens = tokenize_and_filter(doc)
      
      for token in tokens:
        termCounts[token] += 1
      totalToken = sum(termCounts.values())
    
    for term in vocab:
      termCount = termCounts[term]
      condProb = (termCount + 1) / (totalToken + len(vocab))
      condProbList.append([c, term, format(condProb,'.10f')])
  
  return vocab, priorList, condProbList

def writeToTSV(path, priors, likelihoods):
  '''
  Function: 
    write data into the tsv file. Here, data is three columns, with
      first column: prior/likelihood,
      If the line is for a prior, the next columns should be the class name and its prior . 
      If the line is for a likelihood, the next three columns should be the class, the term, and the probability.
  Arguments:
    filename: the name of the file to write to
    priors: a list of [className,prior]
    likelihoods: a list of [classNamem token, conditionalProb]
  Return:
    no return
  '''
  with open(path, 'wt', encoding='utf-8') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for prior in priors:
      className = prior[0]
      priorValue = prior[1]
      tsv_writer.writerow(["prior",className,priorValue])
    for likelihood in likelihoods:
      className = likelihood[0]
      token = likelihood[1]
      likelihoodValue = likelihood[2]
      tsv_writer.writerow(["likelihood",className,token,likelihoodValue])

def main():
  nbcTSVPath, trainJsonPath = getCmdArg()
  isFileExists = check_tsv_file_in_path(nbcTSVPath)
  if isFileExists:
    while True:
      choice =input(f"Are you sure you want to overwrite {nbcTSVPath}? (y/n): ")
      if choice.lower() == "y":
        break
      elif choice.lower() =='n':
        return
      else:
        print("Invalid input, please enter 'y' or 'n'")

  data = getData(trainJsonPath)
  checkFormat(data)
  C,D,dic = getCandD(data)
  V, priorResults, probResults = trainMultinomialNB(C,D,dic)
  
  writeToTSV(nbcTSVPath, priorResults, probResults)

if __name__ == "__main__":
    main()



