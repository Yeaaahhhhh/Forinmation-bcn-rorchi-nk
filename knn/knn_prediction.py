import sys
import ast
import json
import math
from collections import defaultdict

import re
def getCmdArg():
    '''
    Function: This function get the user second input comand line argument. It check whether the input 
    number of command line arguments are correct, whether the file names input order are correct. whether the
    input number is an integer.
    Argument: None
    Return: a tsv file path, and a json path, and the number k
    '''
    try:
        numCommandArg = 4           # For kNN prediction program, the number of command arguments is 3 = 1 python + 2 tsv and test json paths
        if len(sys.argv) != numCommandArg:
            sys.stderr.write('Give more or less command line arguments, make sure the amount of arguments is correct\n')
            sys.exit()


        if ('tsv' not in sys.argv[1]) and ('json' not in sys.argv[2]):
            sys.stderr.write('tsv file or json file not in correct order\n')
            sys.exit()

        kNNTSVPath = str(sys.argv[1])
        testJsonPath = str(sys.argv[2])
        kTheNum = int(sys.argv[3])

    except Exception:
        sys.stderr.write('User input argument error, not found file/folder name\n' +
                         'One of the file names or the number input incorrectly\n')
        sys.exit()
    return kNNTSVPath, testJsonPath, kTheNum

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
 
def euclidean_distance(a, b):
    '''
    Function:
    Calculate the Euclidean distance between two dictionaries of numerical values.
    
    Parameters:
        a (dict): A dictionary of numerical values.
        b (dict): Another dictionary of numerical values.
        
    Returns:
        distance (float): The Euclidean distance between the two dictionaries.
    '''
    common_keys = set(a.keys()) & set(b.keys())
    sqdif = 0
    for key in common_keys:
        sqdif += (a[key]-b[key])**2
    for key in a.keys()-common_keys:
        sqdif += a[key]**2
    for key in b.keys()-common_keys:
        sqdif += b[key]**2
    distance = math.sqrt(sqdif)

    return distance

def find_most_common_element(data):
    '''
    Function:
    Find the most common element in a list of pairs.
    
    Parameters:
    data (list): A list of pairs (score, element).
        
    Returns:
    most_common_element: The element that appears most frequently in the list.
    '''
    counts = {}
    sums = {}
    for score, element in data:
        if element not in counts:
            counts[element] = 0
            sums[element] = 0.0
        counts[element] += 1
        sums[element] += score
    max_count = max(counts.values())
    max_elements = [element for element, count in counts.items() if count == max_count]
    if len(max_elements) == 1:
        return max_elements[0]
    else:
        min_sum = float('inf')
        min_element = None
        for element in max_elements:
            current_sum = sums[element]
            if current_sum < min_sum:
                min_sum = current_sum
                min_element = element
        return min_element

def predict_category(document, vectors):
    '''
    Predict the category of a document based on a set of vector centroids.
    
    Parameters:
        document (dict): A dictionary of numerical values representing the document.
        vectors (dict): A dictionary of vector centroids, where each centroid is a dictionary of numerical values.
        
    Returns:
        predicted_category: The category predicted for the document.
    
    '''
    min_distance = float('inf')
    predicted_category = None
    for category, centroid in vectors.items():
        distance = euclidean_distance(document, centroid)
        if distance < min_distance:
            min_distance = distance
            predicted_category = category

    return predicted_category

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

def knnPredict(tsv_file, test_file, k):
    '''
    Predict the category of documents in a test file using k-nearest neighbors.
    
    Parameters:
        tsv_file (str): The path to a TSV file containing training data in the form of vectors and IDFs.
        test_file (str): The path to a JSON file containing test data.
        k (int): The number of nearest neighbors to use in the prediction.
        
    Returns:
        predicted (dict): A dictionary of document texts and their predicted categories.
    
    '''
    vectors = {}
    idfs = {}
    with open(tsv_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == 'vector':
                if parts[1] not in vectors:
                    temp = []
                    temp.append(ast.literal_eval(parts[2]))
                    vectors[parts[1]] = temp
                else:
                    vectors[parts[1]].append(ast.literal_eval(parts[2]))
            elif parts[0] == 'idf':
                idfs[parts[1]] = float(parts[2])
    # Process test file
    with open(test_file, 'r') as f:
        test_data = json.load(f)
        results = []
        total_weight = 0
        predicted = {}
        for doc in test_data:
            distances = []
            doc_class = doc['category']
            doc_text = doc['text']
            vector = {}
            for term in tokenize_and_filter(doc_text):
                if term in idfs:
                    weight = (1 + math.log(doc_text.count(term))) * idfs[term]
                    vector[term] = weight
                    total_weight += weight**2
            norm = math.sqrt(total_weight)
            vector = {key: value / norm for key, value in vector.items()}

            # distances = {}
            for class_name in vectors.keys():
                for doc_vectors in vectors[class_name]:
                    distance = euclidean_distance(doc_vectors, vector)
                    distances.append((distance,class_name))
            distances.sort()
            distances = distances[:k]  
            category = find_most_common_element(distances)
            predicted[doc_text] = category
    return predicted

def calcSignals(knn_predictions, testData, category):
    '''
    Function:
    Calculates the true positives (TP), false positives (FP), false negatives (FN), true negatives (TN), precision, recall, 
    and F1 score for a given category based on the output of a KNN classifier on a test dataset.
    
    Parameters:
        knn_predictions (dict): A dictionary of document texts and their predicted categories.
        testData (dict): A dictionary of document texts and their true categories.
        category (str): The category for which to calculate performance metrics.
        
    Returns:
        TP, FP, FN, TN, precision, recall, F1 for the given category.
    '''
    TP = 0 
    FP = 0
    FN = 0
    TN = 0
    for doc,predictedClass in knn_predictions.items():
        if doc in testData[category]:
            if category == predictedClass:
                TP += 1
            else:
                FN += 1
        else:
            if category == predictedClass:
                FP += 1
            else:
                TN += 1

    if (TP + FP == 0):
        precision = 0
    else:
        precision = TP / (TP + FP)
    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = (2 * precision * recall) / (precision + recall)

    return TP, FP, FN, TN, precision, recall, F1

def main():
    # start = time.time()
    kNNTSVPath, testJsonPath, kTheNum = getCmdArg()
    data = getData(testJsonPath)
    C, D, dic = getCandD(data) 
    results = knnPredict(kNNTSVPath,testJsonPath,kTheNum)
    F1List = []
    TPtotal = 0
    FPtotal = 0
    FNtotal = 0
    for c in C:
        TP, FP, FN, TN, precision, recall, F1 = calcSignals(results, dic, c)
        F1List.append(F1)
        TPtotal += TP
        FPtotal += FP
        FNtotal += FN
        sys.stdout.write("class: {}, TP: {}, FP: {}, FN: {}, TN: {}, P: {}, R: {}, F1: {}\n".format(c,TP, FP, FN, TN, precision, recall, F1))
    macroAveraged = format(sum(F1List) / len(F1List),'.5f')
    microPrecision = TPtotal / (TPtotal + FPtotal)
    microRecall = TPtotal / (TPtotal + FNtotal)
    microAveraged = format((2 * microPrecision * microRecall) / (microPrecision + microRecall),'.5f')
    sys.stdout.write("macro-averaged F1: {}\n".format(macroAveraged))
    sys.stdout.write("micro-averaged F1: {}\n".format(microAveraged))

    # print(time.time()-start)
if __name__ == "__main__":
    main()