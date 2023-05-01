
# TODOs

1. **SUBMIT** the URL of this repository on eClass. 
2. List the CCID(s) of student(s) working on the project.
3. List all sources consulted while completing the assignment.

|Student name| CCID |
|------------|------|
|student 1   |   xiangtia   |
|student 2   |   qye   |


## kNN analysis

Report the F1 for each class separately, as well as the micro and macro averages for all classes.

|   | k=1 | k=5 | k=11 | k=23 
|---|----|----|----|----|
tech |0.952 |0.930 |0.952 |0.930 |
business |0.980 | 0.960|0.960 |0.960 |
sport | 1.0| 1.0 |0.980 |1.0 |
entertainment |0.950 |0.974 |1.0 |0.974 |
politics |0.976 |1.0 |0.976 |1.0 |
|---|----|----|----|----|
ALL-micro |0.97321 |0.97321 |0.97321 |0.97321 |
ALL-macro |0.97168 |0.97292 |0.97368 | 0.97292|


## Classifier comparison

Report the F1 for each class separately, as well as the micro and macro averages for all classes.

|   | NBC | Rocchio | kNN (k=`11`) 
|---|----|----|----|
tech |0.933 |0.977 |0.952 |
business |0.941 |0.962 | 0.960|
sport |1.0 | 1.0 | 0.980|
entertainment |0.974 |1.0 | 1.0|
politics |0.976 |0.976 | 0.976|
|---|----|----|----|
ALL-micro |0.965 |0.982 |0.97321 |
ALL-macro |0.965 |0.983 | 0.97368|
