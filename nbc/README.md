## REMINDERS

1. Code that does not read all inputs from the command line will get zero for correctness in this assignment.
+ You **MUST** list all consultations you had, including websites you have read while working on this assignment.

### Python NBC Training Program ReadMe

This Python program trains a Naive Bayes Classifier (NBC) using a given dataset in JSON format and writes the training results to a TSV file.

**Python Libraries Used**

    sys,os,json,csv,time,numpy,re,collections
    need to use:
    pip install numpy
    


**Data Structures Used**

    List
    Dictionary
    defaultdict (from collections)

**Algorithms Used**

    Naive Bayes Classifier (Multinomial NB)

**Clean data**
    The program clean data by eliminating all unnecessary punctuation such as ?!<> etc, but keep the dash between the words for example: long-term, previous-installed.
    And remove all stopwords which does not represent meaningful content for current class such as [are,you,me,I,will,do,shall,he,him,she,her......]

**Assumptions**

    Input data is in JSON format and contains a list of dictionaries with "category" and "text" keys.
    The JSON file is well-formed and follows the specified format. Any deviation will result in an error message and program termination.
    The TSV file path is valid and correctly formatted.
    Stop words are predefined and hardcoded within the program if there is not wifi, we cannot download it from nltk online.
    The tsv and json file user input has complete format, that is the 'path+filename+.tsv OR json'

**Error Handling Strategy**

    The program checks for the correct number of command-line arguments and valid file paths. If any discrepancies are found, a relevant error message is displayed, and the program terminates.
    The JSON data is checked for proper formatting. If any issues are detected, a corresponding error message is shown, and the program exits.
    The program checks if the TSV file already exists in the specified path. If it does, the user is prompted to confirm whether they want to overwrite the existing file.
    Exceptions are handled using try-except blocks, and appropriate error messages are displayed.

**Usage**
**Command Line Arguments**

    The first argument is the path to the input JSON file containing the training data.
    The second argument is the path to the output TSV file, where the training results will be written.

**Example**
Assume the current file path is w23-hw3-Yeaaahhhhh(main)

**The example of execution is:**
    python3 nbc/nbc_train.py ./data/train.json ./nbc_model.tsv

This command trains an NBC using the data in ./data/train.json and writes the training results to ./nbc_model.tsv


## Naive Bayes Classifier (NBC) for text classification

This program is a Python implementation of the Naive Bayes Classifier (NBC) algorithm used for text classification. The NBC algorithm is a probabilistic method that applies Bayes' theorem, with the assumption of independence between features, to classify documents based on their text contents. In this program, NBC is applied to a set of training documents to estimate the prior and conditional probabilities, which are then used to classify a set of test documents into predetermined categories.
**Data structure used**

    Counter: A dictionary subclass that counts occurrences of items in a list.
    defaultdict: A dictionary subclass that provides a default value for a nonexistent key.
    numpy array: A multidimensional array object that provides efficient numerical computations.
    json: A lightweight data interchange format that is easy to read and write.
    re: regmex to tokenize and filter the string

**Algorithms used**

    Naive Bayes Classifier: a probabilistic algorithm that uses Bayes' theorem with the assumption of independence between features.

**Assumptions**

    The text contents of the documents are assumed to be independent features.
    The text contents of the documents are preprocessed to remove stop words and punctuation marks.
    The training and test documents are assumed to be in the TSV and JSON formats, respectively.
    The tsv and json file user input has complete format, that is the 'path+filename+.tsv OR json'

**Error handling strategies**

    The program checks the number and order of the command line arguments to ensure that the TSV and JSON files are provided in the correct order.
    The program uses try-except blocks to catch and handle file opening and input errors.

**Example of execution of the program**

python3 nbc/nbc_prediction.py ./nbc_model.tsv ./data/test.json

The program takes two arguments from the command line: 
    the path to the TSV file containing the training data and the path to the JSON file containing the test data. 
    The program then applies NBC to the training data to calculate the prior and conditional probabilities, and uses these probabilities to classify the test data into predetermined categories. The results are then printed to the console.
    
**References**
1. https://www.geeksforgeeks.org/python-os-listdir-method/
