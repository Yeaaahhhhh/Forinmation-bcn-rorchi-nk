## REMINDERS

1. Code that does not read all inputs from the command line will get zero for correctness in this assignment.
+ You **MUST** list all consultations you had, including websites you have read while working on this assignment.

## kNN Training Program

This repository contains a kNN training program written in Python. The purpose of this program is to train a k-Nearest Neighbors (kNN) classifier using given training data and save the data structures needed for classification as a TSV file.
Features

 **Program Functionality**: 
   The program takes in a JSON file containing training data and generates a TSV file with the necessary data structures for classification.
   
 **Data Structures**: The program uses dictionaries, lists, and defaultdict from the collections module to store and manipulate data.
 
 **Libraries Used**: 
 This program uses the following libraries:
     json
     csv
     sys
     os
     math
     collections
     collections.defaultdict
     
 **Algorithms Used**: The program implements the following algorithms:
 
     Term Frequency (TF) calculation
     Document Frequency (DF) calculation
     Inverse Document Frequency (IDF) calculation
     Term frequency-inverse document frequency (TF-IDF) weighting
     Vector normalization

**Clean data**
    The program clean data by eliminating all unnecessary punctuation such as ?!<> etc, but keep the dash between the words for example: long-term, previous-installed.
    And remove all stopwords which does not represent meaningful content for current class such as [are,you,me,I,will,do,shall,he,him,she,her......]

 **Assumptions**: 
    The program assumes that the input JSON file is well-formatted and contains the appropriate keys, such as "category" and "text".
    Each category should have a text that represent current doument class.
    Each text should have a text that represent the content of the document.
    Stopwords are hardcoded in case that there is no wifi to download it from nltk module.
    The tsv and json file user input has complete format, that is the 'path+filename+.tsv OR json'
 **Error Handling Strategies**: The program contains error handling for:
     Incorrect command line arguments
     Invalid file paths and file names
     Incorrect JSON file format
     
 **Example of Execution**: To execute the program, run the following command in the terminal (assume current file path is w23...Yeaaahhhhh)
 
    python3 knn/knn_create_vectors.py ./data/train.json ./knn_model.tsv

Replace ./data/train.json with the path to your JSON training data file and ./knn_model.tsv with the path where you want the TSV file to be saved.



## kNN Prediction Algorithm
**Introduction**

This is a Python implementation of kNN prediction algorithm for text classification. Given a set of training documents, this program uses kNN algorithm to classify test documents based on their similarity to the training documents.
**Data Structure Used**

The training documents are stored as a TSV file where each row represents a document and each column represents a term frequency. The test documents are stored as a JSON file with each object containing the text of the document and the true category.
**Library Used**

    sys : For handling command-line arguments and error messages
    ast : For parsing the string representation of lists and dictionaries from TSV file
    json : For parsing the JSON file containing test documents
    math : For computing the square root of a number
    defaultdict : For creating a dictionary that returns a default value when a non-existing key is accessed.
    re: regmex to tokenize and filter the string
**Algorithms Used**

    kNN (k-Nearest Neighbors) : This algorithm uses Euclidean distance to find the k nearest neighbors of a test document from the training documents. The category of the test document is then predicted as the most frequent category among the k neighbors.

**Clean data**
    The program clean data by eliminating all unnecessary punctuation such as ?!<> etc, but keep the dash between the words for example: long-term, previous-installed.
    And remove all stopwords which does not represent meaningful content for current class such as [are,you,me,I,will,do,shall,he,him,she,her......]

**Assumptions**

 The TSV file containing the training documents has the following format:
     The first column is 'vector' and the second column is the category of the document
     The third column is a dictionary that maps each term to its term frequency in the document
     Each row represents a document
 The JSON file containing the test documents has the following format:
     Each object has two key-value pairs: 'category' and 'text'
     'category' is the true category of the document
     'text' is the text content of the document
The tsv and json file user input has complete format, that is the 'path+filename+.tsv OR json'
**Error Handling Strategies**

    If the number of command-line arguments is not 4, an error message is displayed.
    If the input file names are not in the correct order or the input number is not an integer, an error message is displayed.
    If an inappropriate argument is passed, the program exits with an error message.

**Example of Execution**

To run the program, execute the following command:

   python knn_prediction.py <tsv_file_path> <test_json_path> k

where <tsv_file_path> is the path to the TSV file containing the training documents, <test_json_path> is the path to the JSON file containing the test documents, and k is the number of nearest neighbors to consider.

For example:

   python3 knn/knn_prediction.py ./knn_model.tsv ./data/test.json 5

This command classifies the test documents in test.json using the training documents in train.tsv with k=5.

**References**
https://www.aipython.in/python-literal_eval/
