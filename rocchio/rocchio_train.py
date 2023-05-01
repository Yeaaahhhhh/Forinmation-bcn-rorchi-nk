import json
import math
from collections import defaultdict
from typing import List, Dict
import sys
import os
import re
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


        if ('tsv' not in sys.argv[2]) and ('json' not in sys.argv[1]):
            sys.stderr.write('tsv file or json file not in correct order\n')
            sys.exit()

        trainJsonPath = str(sys.argv[1])
        rocchioTSVPath = str(sys.argv[2])

    except Exception:
        sys.stderr.write('User input argument error, not found file/folder name\n' +
                         'One of the file names input incorrectly\n')
        sys.exit()
    return  trainJsonPath, rocchioTSVPath

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

def check_tsv_file_in_path(file_path):
    '''
    This function serves for training program write to tsv that whether or not overwrite a file,
    It checks if the folder contains the file name, return True, else false, for later 
    write to tsv file function use.

    Argument: file path user input, that is sys.argv[2]

    return: boolean value
    '''
    path_parts = file_path.split("/")
    path_to_file = "/".join(path_parts[:-1])
    tsv_file_name = path_parts[-1]
    for file_name in os.listdir(path_to_file):
        if file_name.endswith(".tsv") and file_name == tsv_file_name:
            return True
    return False

def read_json_file(file_path: str) -> List[Dict[str, str]]:
    '''
    Function: Reads a JSON file from the given file path and returns its data as a list of dictionaries.
    
    Parameters:
    file_path: A string that represents the file path of the JSON file to be read.
    
    Returns:
    A list of dictionaries that contain the data from the JSON file.
    '''
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
    except Exception:
        sys.stderr.write("no such file or directory, please try again")
        sys.exit()
    return data
def calculate_tf_idf_and_centroids(data: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    '''
    Function: Calculates the term frequency-inverse document frequency (tf-idf) weights and centroids 
    for each category from the given list of dictionaries.
    
    Parameters:
    data: A list of dictionaries where each dictionary represents a document with its text and category.
    
    Returns:
    A dictionary of dictionaries where each key represents a category and its value is another dictionary that 
    contains the tf-idf weights for each term in that category, as well as a dictionary that contains the idf 
    values for each term in the entire dataset.
    '''
    # Initialize variables
    N = len(data)
    class_counts = defaultdict(int)
    class_centroids = defaultdict(lambda: defaultdict(float))
    term_doc_freq = defaultdict(int)
    term_idf = defaultdict(float)

    # Calculate term frequencies and document frequencies
    for doc in data:
        category = doc["category"]
        text = doc["text"]
        class_counts[category] += 1

        term_freq = defaultdict(int)
        for term in tokenize_and_filter(text):
            term_freq[term] += 1

        # Update document frequency
        for term in set(term_freq.keys()):
            term_doc_freq[term] += 1

    # Calculate inverse document frequencies
    for term, doc_freq in term_doc_freq.items():
        term_idf[term] = math.log2(N / doc_freq)

    # Calculate tf-idf weights and centroids
    for doc in data:
        category = doc["category"]
        text = doc["text"]

        term_freq = defaultdict(int)
        for term in tokenize_and_filter(text):
            term_freq[term] += 1

        # Compute tf-idf weights and normalize using cosine normalization
        doc_vector = defaultdict(float)
        vector_sum = 0
        for term, freq in term_freq.items():
            tf = 1 + math.log(freq)
            idf = term_idf[term]
            weight = tf * idf
            vector_sum += weight ** 2
            doc_vector[term] = weight

        vector_norm = math.sqrt(vector_sum)
        for term, weight in doc_vector.items():
            normalized_weight = weight / vector_norm
            class_centroids[category][term] += normalized_weight

    # Compute final centroid values
    for category, centroid in class_centroids.items():
        for term, weight_sum in centroid.items():
            centroid[term] = weight_sum / class_counts[category]

    return class_centroids, term_idf

def write_tsv_file(file_path: str, centroids: Dict[str, Dict[str, float]], term_idf: Dict[str, float]):
    '''
    Function: Writes the calculated centroids and idf values to a TSV file at the given file path.
    
    Parameters:
    file_path: A string that represents the file path of the TSV file to be written.
    centroids: A dictionary of dictionaries where each key represents a category and its value is another dictionary that contains the tf-idf weights for each term in that category.
    term_idf: A dictionary that contains the idf values for each term in the entire dataset.
    
    Returns:
    None
    '''
    with open(file_path, "w") as f:
        # Write centroid lines
        for category, centroid in centroids.items():
            centroid_str = json.dumps(centroid)
            f.write(f"centroid\t{category}\t{centroid_str}\n")

        # Write idf lines
        for term, idf in term_idf.items():
            f.write(f"idf\t{term}\t{idf:.7f}\n")

def main():
    json_file_path = getCmdArg()[0]
    tsv_file_path = getCmdArg()[1]
    
    isFileExists = check_tsv_file_in_path(tsv_file_path)
    if isFileExists:
        while True:
            choice =input(f"Are you sure you want to overwrite {tsv_file_path}? (y/n): ")
            if choice.lower() == "y":
                break
            elif choice.lower() =='n':
                return
            else:
                print("Invalid input, please enter 'y' or 'n'")
    
    data = read_json_file(json_file_path)
    checkFormat(data)
    centroids, term_idf = calculate_tf_idf_and_centroids(data)
    write_tsv_file(tsv_file_path, centroids, term_idf)

if __name__ == "__main__":
    main()