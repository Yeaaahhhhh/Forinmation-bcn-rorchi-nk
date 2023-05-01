import sys
import json
import math
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
        numCommandArg = 3           # For rocchio prediction program, the number of command arguments is 3 = 1 python + 2 tsv and test json paths
        if len(sys.argv) != numCommandArg:
            sys.stderr.write('Give more or less command line arguments, make sure the amount of arguments is correct\n')
            sys.exit()


        if ('tsv' not in sys.argv[1]) and ('json' not in sys.argv[2]):
            sys.stderr.write('tsv file or json file not in correct order\n')
            sys.exit()

        rocchioTSVPath = str(sys.argv[1])
        testJsonPath = str(sys.argv[2])

    except Exception:
        sys.stderr.write('User input argument error, not found file/folder name\n' +
                         'One of the file names input incorrectly\n')
        sys.exit()
    return rocchioTSVPath, testJsonPath

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

def load_data(json_file, tsv_file):
    '''
    Function: This function loads data from a JSON file and a TSV file and returns the loaded documents, centroids, and idf values.
    
    Parameters:
    json_file (str): The path of the JSON file to load data from.
    tsv_file (str): The path of the TSV file to load data from.

    Returns: 
    A tuple containing:
        documents (list): A list of dictionaries representing the loaded documents.
        centroids (dict): A dictionary containing category names as keys and centroid vectors as values.
        idf (dict): A dictionary containing terms as keys and idf values as values.
    '''
    with open(json_file, 'r') as f:
        documents = json.load(f)

    centroids = {}
    idf = {}
    with open(tsv_file, 'r') as f:
        for line in f:
            columns = line.strip().split('\t')
            if columns[0] == 'centroid':
                centroids[columns[1]] = json.loads(columns[2])
            elif columns[0] == 'idf':
                idf[columns[1]] = float(columns[2])

    return documents, centroids, idf

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

def tf_idf_weights(document, idf):
    '''
    Function: This function calculates the TF-IDF weights for a given document and idf values.
    
    Parameters:
    document (str): The text of the document to calculate weights for.
    idf (dict): A dictionary containing terms as keys and idf values as values.
    
    Returns: 
    A tuple containing:
        weights (dict): A dictionary containing terms as keys and TF-IDF weights as values.
        document (str): The text of the input document.
    '''
    tf = defaultdict(int)
    for term in tokenize_and_filter(document):
        tf[term] += 1

    weights = {}
    for term, freq in tf.items():
        if term in idf:
            weights[term] = (1 + math.log(freq)) * idf[term]

    return weights,document

def normalize(weights):
    '''
    Function: This function normalizes a given weight vector.
    
    Parameters:
    weights (dict): A dictionary containing terms as keys and weight values as values.
    
    Returns: A normalized weight vector (dict).
    '''
    norm = math.sqrt(sum(w ** 2 for w in weights.values()))
    for term, weight in weights.items():
        weights[term] = weight / norm
    return weights

def euclidean_distance(a, b):
    '''
    Function: This function calculates the Euclidean distance between two vectors.
    
    Parameters:
    a (dict): A dictionary containing terms as keys and weight values as values for the first vector.
    b (dict): A dictionary containing terms as keys and weight values as values for the second vector.
    
    Returns: The Euclidean distance (float) between the two input vectors.
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

def predict_category(document, centroids):
    '''
    Functionality: This function predicts the category of a given document based on the nearest centroid.
    
    Parameters:
    document (dict): A dictionary containing terms as keys and weight values as values for the input document.
    centroids (dict): A dictionary containing category names as keys and centroid vectors as values.
    
    Returns: The predicted category (str) for the input document.
    '''
    min_distance = float('inf')
    predicted_category = None
    for category, centroid in centroids.items():
        distance = euclidean_distance(document, centroid)
        if distance < min_distance:
            min_distance = distance
            predicted_category = category

    return predicted_category

def classify_documents(json_file, tsv_file):
    '''
    Functionality: This function classifies a list of documents based on pre-calculated centroids and idf values.
    
    Parameters:
    json_file (str): The path of the JSON file containing the documents to classify.
    tsv_file (str): The path of the TSV file containing the centroids and idf values.
    
    Returns: A list of dictionaries, where each dictionary contains the input document and its predicted category.
    '''
    documents, centroids, idf = load_data(json_file, tsv_file)
    predictions = []
    for doc in documents:
        true_category = doc['category']
        text = doc['text']
        weights,document = tf_idf_weights(text, idf)
        normalized_weights = normalize(weights)
        predicted_category = predict_category(normalized_weights, centroids)
        predictions.append({document:predicted_category})
    return predictions

def calcSignals(rocchio_predictions, testData, category):
    """
    Calculate evaluation signals (TP, FP, FN, TN, precision, recall, and F1) for a given category.

    Parameters:
        - rocchio_predictions (list): A list of predictions generated by the Rocchio algorithm.
        - testData (dict): A dictionary containing the test data, where the keys are the category names and
                           the values are lists of documents in that category.
        - category (str): The name of the category for which to calculate the evaluation signals.

    Returns:
        - TP (int): The number of true positives.
        - FP (int): The number of false positives.
        - FN (int): The number of false negatives.
        - TN (int): The number of true negatives.
        - precision (float): The precision score.
        - recall (float): The recall score.
        - F1 (float): The F1 score.
    """
    TP = 0 
    FP = 0
    FN = 0
    TN = 0
    for prediction in rocchio_predictions:
        for doc,predictedClass in prediction.items():
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
    rocchioTSVPath, testJsonPath = getCmdArg()
    data = getData(testJsonPath)
    checkFormat(data)
    C, D, dic = getCandD(data) 
    results = classify_documents(testJsonPath, rocchioTSVPath)
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

if __name__ == "__main__":
    main()