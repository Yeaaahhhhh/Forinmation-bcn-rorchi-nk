import sys
import math
import json
import re
import numpy as np
from collections import Counter, defaultdict
from nbc_train import getData, getCandD, checkFormat


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

        if ('tsv' not in sys.argv[1]) and ('json' not in sys.argv[2]):
            sys.stderr.write('tsv file or json file not in correct order\n')
            sys.exit()

        nbcTSVPath = str(sys.argv[1])
        testJsonPath = str(sys.argv[2])

    except Exception:
        sys.stderr.write('User input argument error, not found file/folder name\n' +
                         'One of the file names input incorrectly\n')
        sys.exit()
    return nbcTSVPath, testJsonPath

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
    "wouldn't", 'above', 'will', 'that', 'again', 'shan', 'what', 'which', "shouldn't", 'my', 'our', 'did', 'was', 'don','.',',','?','!'}
    
    # Filter out stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    return filtered_tokens

def get_common_words(vocabularies, input_string):
    '''
    Function:
    Finds the common words between the given list of vocabularies and a given input string.
    Returns a list of common words.

    Parameters:
    vocabularies (list): List of words to compare against the input string.
    input_string (str): Input string to compare against the list of vocabularies.

    Returns:
    common_words (list): List of common words between the list of vocabularies and the input string.
    '''

    vocab_array = np.array(vocabularies)
    input_array = np.array(tokenize_and_filter(input_string))
    common_words = np.intersect1d(vocab_array, input_array)
    return common_words.tolist()




def extractVolcabularyFromDoc(V, doc):
    '''
    Function:
    extract vocabularies from a list of strings
    Argument:
    doc: a list of strings
    Return:
    vocabularies: a list of vocabularies
    '''
    tokens = tokenize_and_filter(doc)
    common_words = get_common_words(V, ' '.join(tokens))
    return common_words

def getPriorAndCondProb(tsvfile):
    '''
    Parses a TSV file containing prior and conditional probabilities for a Naive Bayes model.
    Returns a dictionary of prior probabilities, a nested dictionary of conditional probabilities, and a list of vocabularies.

    Parameters:
    tsvfile (str): Path to the TSV file containing prior and conditional probabilities.

    Returns:
    priors (dict): Dictionary of prior probabilities for each class.
    conditionalProbs (dict): Nested dictionary of conditional probabilities for each vocabulary term and class.
    vocabularies (list): List of vocabulary terms in the order they appear in the TSV file.
    '''
    priors = {}
    conditionalProbs = {}
    vocabularies = []
    try:
        with open(tsvfile, encoding='utf-8') as files:
            for line in files:
                columns = line.strip().split("\t")
                if columns[0] == "prior":
                    priors[columns[1]] = float(columns[2])
                elif columns[0] == "likelihood":
                    vocabularies.append(columns[2])
                    if columns[2] not in conditionalProbs:
                        conditionalProbs[columns[2]] = {columns[1]:float(columns[3])}
                    else:
                        conditionalProbs[columns[2]][columns[1]] = float(columns[3])
    except Exception as e:
        sys.stderr.write('file opened error, please try again\n')
        sys.exit()
    return priors, conditionalProbs, vocabularies

def applyMultinomialNB(C, V, prior, condprob, d):
    '''
    Applies a Multinomial Naive Bayes classifier to a document `d` and returns the result.

    Parameters:
    - C (list): A list of class labels.
    - V (list): A list of vocabularies.
    - prior (dict): A dictionary of prior probabilities for each class.
    - condprob (dict): A dictionary of conditional probabilities for each vocabulary given a class.
    - d (str): A document to be classified.

    Returns:
    - result (list): A list containing the input document, the class with the highest score, and the highest score itself.
    '''

    words = extractVolcabularyFromDoc(V, d)
    scores = {}
    for c in C:
        scores[c] = math.log2(prior[c])
        for t in words:
            scores[c] += math.log2(condprob[t][c])
    maxClass = max(scores, key=scores.get)
    maxScore = scores[maxClass]
    return [d,maxClass,maxScore]

def getAllNBs(tsvfile, docs, classes):
    '''
    Applies a Multinomial Naive Bayes classifier to a list of documents `docs` using the prior probabilities and 
    conditional probabilities in the TSV file `tsvfile`, and returns a list of results.

    Parameters:
    - tsvfile (str): A path to the TSV file containing the prior probabilities and conditional probabilities.
    - docs (list): A list of documents to be classified.
    - classes (list): A list of class labels.

    Returns:
    - results (list): A list of results, where each result is a list containing the input document, the class with 
    the highest score, and the highest score itself.
    '''

    priors, condProbs, V = getPriorAndCondProb(tsvfile)
    results = []
    for d in docs:
        result = applyMultinomialNB(classes, V, priors, condProbs, d)
        results.append(result)
    return results

def calcSignals(nbList, testData, category):
    '''
    Function:
    Calculates the true positives (TP), false positives (FP), false negatives (FN), true negatives (TN), precision, recall, 
    and F1 score for a given category based on the output of a Naive Bayes classifier on a test dataset.
    
    Parameters:
    nbList (list): The output of a Naive Bayes classifier on a test dataset, in the form of a list of tuples (document, predictedClass, score).
    testData (dict): A dictionary containing the test data, where each key represents a category and its value is a list of documents belonging to that category.
    category (str): The category for which to calculate the signals (e.g., "positive", "negative").
    
    Returns:
    TP (int): The number of true positives.
    FP (int): The number of false positives.
    FN (int): The number of false negatives.
    TN (int): The number of true negatives.
    precision (float): The precision score.
    recall (float): The recall score.
    F1 (float): The F1 score.
    '''
    
    TP = 0 
    FP = 0
    FN = 0
    TN = 0
    for doc, predictedClass, score in nbList:
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
    nbcTSVPath, testJsonPath = getCmdArg()
    data = getData(testJsonPath)
    checkFormat(data)
    C, D, dic = getCandD(data) 
    results = getAllNBs(nbcTSVPath, D, C)
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

    # calculations for macro and micro averaged
    macroAveraged = format(sum(F1List) / len(F1List),'.5f')
    microPrecision = TPtotal / (TPtotal + FPtotal)
    microRecall = TPtotal / (TPtotal + FNtotal)
    microAveraged = format((2 * microPrecision * microRecall) / (microPrecision + microRecall),'.5f')
    sys.stdout.write("macro-averaged F1: {}\n".format(macroAveraged))
    sys.stdout.write("micro-averaged F1: {}\n".format(microAveraged))
    
if __name__ == "__main__":
    main()