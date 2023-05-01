## REMINDERS

1. Code that does not read all inputs from the command line will get zero for correctness in this assignment.
+ You **MUST** list all consultations you had, including websites you have read while working on this assignment.

## Rocchio training Program Readme

This Python program is a text classifier based on the Rocchio algorithm. It takes as input a JSON file containing training data and a TSV file containing test data. The program outputs a TSV file with the classification results for each test data point.

**Data structure used**

The program uses the following data structures:

    List[Dict[str, str]]: to store the training and test data. Each element in the list is a dictionary containing two keys: "category" and "text".
    Dict[str, Dict[str, float]]: to store the class centroids computed during training.
    Dict[str, float]: to store the IDF values for each term.

**Library used**

The program uses the following Python libraries:

    json: to read the input JSON file.
    math: to compute logarithms and square roots.
    collections: to use defaultdicts to simplify the code.
    sys: to write error messages and exit the program in case of incorrect user input.
    os: to check if the TSV file for the output already exists.
    re: regmex to tokenize and filter the string

**Algorithms used**

The program uses the following algorithms:

    Rocchio algorithm: to classify the test data based on the training data.
    TF-IDF weighting: to assign weights to the terms in the documents.
    Cosine normalization: to normalize the TF-IDF vectors.

**Assumptions**

The program assumes that:

    The input JSON file has the correct format, i.e., each document has a "category" and "text" key with non-empty string values.
    The input TSV file has the correct format, i.e., each row has a non-empty string value for the text to be classified.
    The input TSV file for the output does not already exist in the folder.
    The tsv and json file user input has complete format, that is the 'path+filename+.tsv OR json'
Note that the centroid for each class in the tsv file is pretty large, it shows the centroid for one class in the third column, did not occupy other spaces or columns, the format is totally OK.

**Error handling strategies**

The program handles the following errors:

    Incorrect number of command line arguments: the program expects 3 arguments: the Python script, the path to the JSON file, and the path to the TSV file for the test data. If the number of arguments is different, an error message is printed and the program exits.
    Incorrect order of command line arguments: the program assumes that the JSON file is the first argument and the TSV file is the second argument. If the order is different, an error message is printed and the program exits.
    Incorrect format in the JSON file: if a document does not have exactly two keys ("category" and "text") or if one of the keys has an empty string value, an error message is printed and the program exits.
    File not found: if one of the file paths provided by the user is incorrect, an error message is printed and the program exits.
    TSV file for output already exists: if the TSV file for the output already exists in the folder, an error message is printed and the program exits. **OR** check whether to overwrite it, user can input y or n to confirm. If not overwriting then exit.

**Example of execution of the program**
Assume the current file path is on w23-hw3-Yeaaahhhhh
The example execution:
    python3 rocchio/rocchio_train.py ./data/train.json ./rocchio_model.tsv

This command reads the training data from the train_data.json file and outputs the classification results to a new TSV file. 
If an error occurs, an error message is printed to the console and the program exits.





## Rocchio Prediction Algorithm

This program implements the Rocchio Prediction algorithm for document classification, which is based on the Rocchio algorithm for relevance feedback. The algorithm classifies documents by comparing their tf-idf representation to the centroids of the precalculated classes.
1. **What does the program do?**

    The program takes in a TSV file containing precomputed class centroids and a JSON file containing test documents. It computes the tf-idf representation of the       test documents and classifies them into the given classes using the Euclidean distance metric. The program then calculates the performance metrics, such as           precision, recall, and F1-score for each class and reports the macro-averaged and micro-averaged F1-scores.

2. **Libraries Used**

    sys
    json
    math
    collections
    re
    
3. **Data Structures Used**

    Lists
    Dictionaries
    Default dictionaries (from collections)

4. **Assumptions**

    The input TSV file contains class centroids and idf values in the correct format.
    The input JSON file contains test documents in the correct format, with keys "category" and "text".
    The text data in the JSON file is non-empty.
    The tsv and json file user input has complete format, that is the 'path+filename+.tsv OR json'
5. **Error Handling Strategies**

    The program checks if the number of command line arguments is correct and if the file names are in the correct order.
    The program checks if the JSON file has the correct format, with keys "category" and "text".
    Exceptions are handled gracefully, with error messages written to stderr and the program exiting with an appropriate code.

6. **Example of Execution of the Program**

Execute the program with the following command:

python3 rocchio_prediction.py ./rocchio_model.tsv ./data/test.json

Replace <path_to_tsv_file> with the path of the TSV file containing precomputed class centroids and idf values, and <path_to_json_file> with the path of the JSON file containing test documents.

The program will print the class-wise performance metrics and the macro-averaged and micro-averaged F1-scores to the stdout.
