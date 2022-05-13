# Brown Build code 

This projects aims at identifying brown builds (unreliable builds) from the CI 
build jobs. In this folder, you'll find the source code to extract words from 
build jobs' log files and to create process the extracted vocabulary and 
classify the jobs using a XGBoost model.

## Requirements

To run the scripts, you need to have Golang installed and Python 3.

For Python, the requirements are provided in the file `requirements.txt`.
Run the following line for installing the requirements:

```
pip install -r requirements.txt
```


## Dataset
To be able to run the script, you should have a folder containing job logs with 
title:

{builddate}\_{buildid}\_{commitid}\_{classification}\_{buildname}.log 

- {builddate} is the date at which the build was started in the following format YYYY\_MM\_DD\_HH\_MM\_SS 
- {buildid} is the id of the build job
- {commitid} is the cl / commit hash that was built 
- {classification} shows if the build failed (1) or succeeded (0) 
- {buildname} is the name of the build job

A dataset already scraped is provided with this project. You can find it under 
`graphviz/`. Five zip are provided and all the job logs of those 5 zips should be 
put in a same directory, for example, in `./dataset/graphviz/`.

Caution: unzipped, `graphviz.zip` contains 37GB of data.

## Vocabulary extraction
The vocabulary extraction is done using the `main_extract.go` file. The 
command line to use is the following:

```
go run main_extract.go -proc 5 -path ./dataset/graphviz/ -out ./dataset/graphviz_extracted/
```

Output:

```
Done ./dataset/graphviz_extracted/
---  1h22m30.5163144s  ---
```
## Model creation and evaluation

### Simple cross validation run 
To create the brown build detection prototype and test it on your dataset using 
cross validation, use the following command line. The example is given using 
the graphviz dataset.

```
python main_process.py -d ./dataset/graphviz_extracted/
```

The output should look like this:

Output:

```
Experiment: {'path_data': './dataset/graphviz_extracted/'}
Load  experiments/default/data.p ... (computing) ...Done in 374.79 sec
Load  experiments/default/sets.p ... (computing) ...Done in 12.12 sec
Load  experiments/default/vectors.p ... (computing) ...Done in 121.72 sec
Run          | F1-Score     Precision    Recall       Specificity  |
--------------------------------------------------------------------
RANDOM50     | 10.4         13.1         50.0         50.0         |
RANDOMB      | 6.5          13.1         13.1         86.9         |
ALWAYSBROWN  | 11.6         13.1         100          0            |
XGB          | 60.0         60.0         60.0         96.7         |
===== TOTAL TIME:  509.37 sec =====
```


### 10fold cross validation run # Brown Build code 

This projects aims at identifying brown builds (unreliable builds) from the CI 
build jobs. In this folder, you'll find the source code to extract words from 
build jobs' log files and to create process the extracted vocabulary and 
classify the jobs using a XGBoost model.

## Requirements

To run the scripts, you need to have Golang installed and Python 3.

For Python, the requirements are provided in the file `requirements.txt`.
Run the following line for installing the requirements:

```
pip install -r requirements.txt
```


## Dataset
To be able to run the script, you should have a folder containing job logs with 
title:

{builddate}\_{buildid}\_{commitid}\_{classification}\_{buildname}.log 

- {builddate} is the date at which the build was started in the following format YYYY\_MM\_DD\_HH\_MM\_SS 
- {buildid} is the id of the build job
- {commitid} is the cl / commit hash that was built 
- {classification} shows if the build failed (1) or succeeded (0) 
- {buildname} is the name of the build job

A dataset already scraped is provided with this project. You can find it under 
`dataset/graphviz.zip`. Extract the zip in the dataset directory.

## Vocabulary extraction
The vocabulary extraction is done using the `main_extract.exe` file. The 
command line to use is the following:

```
./main_extract.exe -proc 5 -path ./dataset/graphviz/ -out ./dataset/graphviz_extracted/
```

This executable was build on a Windows machine, so if it is not working for you 
machine or if you want to run the source code itself, the go file is also 
provided. To run the go file, use the following command line:
```
go run main_extract.go -proc 5 -path ./dataset/graphviz/ -out ./dataset/graphviz_extracted/
```

Output (in both cases):

```
Done ./dataset/graphviz_extracted/
---  1h22m30.5163144s  ---
```
## Model creation and evaluation

### Simple cross validation run 
To create the brown build detection prototype and test it on your dataset using 
cross validation, use the following command line. The example is given using 
the graphviz dataset.

```
python main_process.py -d ./dataset/graphviz_extracted/
```

The output should look like this:

Output:

```
Experiment: {'path_data': './dataset/graphviz_extracted/'}
Load  experiments/default/data.p ... (computing) ...Done in 374.79 sec
Load  experiments/default/sets.p ... (computing) ...Done in 12.12 sec
Load  experiments/default/vectors.p ... (computing) ...Done in 121.72 sec
Run          | F1-Score     Precision    Recall       Specificity  |
--------------------------------------------------------------------
RANDOM50     | 10.4         13.1         50.0         50.0         |
RANDOMB      | 6.5          13.1         13.1         86.9         |
ALWAYSBROWN  | 11.6         13.1         100          0            |
XGB          | 60.0         60.0         60.0         96.7         |
===== TOTAL TIME:  509.37 sec =====
```


### 10fold cross validation run 
If you want to use the 10fold cross validation as shown in the paper, use:

```
python main_process.py -d ./dataset/graphviz_extracted/ --10fold
``` 

Output:

```
Experiment: {'path_data': './dataset/graphviz_extracted/'}
Load  experiments/default/data.p ... (computing) ...Done in 375.12 sec
Load  experiments/default/sets_10fold.p ... (computing) ...Done in 119.29 sec
Load  experiments/default/vectors_10fold_run1_turn1.p ... (computing) ...Done in 106.1 sec
Load  experiments/default/vectors_10fold_run1_turn2.p ... (computing) ...Done in 107.34 sec
Load  experiments/default/vectors_10fold_run2_turn1.p ... (computing) ...Done in 146.73 sec
Load  experiments/default/vectors_10fold_run2_turn2.p ... (computing) ...Done in 145.48 sec
Load  experiments/default/vectors_10fold_run3_turn1.p ... (computing) ...Done in 140.68 sec
Load  experiments/default/vectors_10fold_run3_turn2.p ... (computing) ...Done in 140.43 sec
Load  experiments/default/vectors_10fold_run4_turn1.p ... (computing) ...Done in 133.3 sec
Load  experiments/default/vectors_10fold_run4_turn2.p ... (computing) ...Done in 135.65 sec
Load  experiments/default/vectors_10fold_run5_turn1.p ... (computing) ...Done in 122.81 sec
Load  experiments/default/vectors_10fold_run5_turn2.p ... (computing) ...Done in 123.76 sec
Load  experiments/default/vectors_10fold_run6_turn1.p ... (computing) ...Done in 141.75 sec
Load  experiments/default/vectors_10fold_run6_turn2.p ... (computing) ...Done in 143.56 sec
Load  experiments/default/vectors_10fold_run7_turn1.p ... (computing) ...Done in 128.52 sec
Load  experiments/default/vectors_10fold_run7_turn2.p ... (computing) ...Done in 125.67 sec
Load  experiments/default/vectors_10fold_run8_turn1.p ... (computing) ...Done in 155.09 sec
Load  experiments/default/vectors_10fold_run8_turn2.p ... (computing) ...Done in 157.36 sec
Load  experiments/default/vectors_10fold_run9_turn1.p ... (computing) ...Done in 144.89 sec
Load  experiments/default/vectors_10fold_run9_turn2.p ... (computing) ...Done in 143.88 sec
Load  experiments/default/vectors_10fold_run10_turn1.p ... (computing) ...Done in 131.99 sec
Load  experiments/default/vectors_10fold_run10_turn2.p ... (computing) ...Done in 138.7 sec
Run          | F1-Score     Precision    Recall       Specificity  |
--------------------------------------------------------------------
RANDOM50     | 10.4         13.1         50.0         50.0         |
RANDOMB      | 6.5          13.1         13.1         86.9         |
ALWAYSBROWN  | 11.6         13.1         100          0            |
XGB          | 51.6         46.9         57.3         90.5         |
===== TOTAL TIME:  3317.96 sec =====
```

### Additional parameters
Additional parameters are available to choose the Experiment set-up, which must be added after `python main_process.py`:
- `-d <str>` / `--path_data <str>`: [mandatory] 

  Path to the extracted dataset.
  
- `--setting_name <str>`: [optional]

  Setting name. Will be used as directory name to save the pickle files. 
  (Default= 'default')
  
- `--ngram <list>`: [optional]

  List of values N to consider (only values 1 and 2 are available with this extraction set-up)
  (Default= [2])
  
- `--oversampling <bool>`: [optional]

  Bool value. If True, the training set is oversampled.
  (Default= True)
  
- `--fail_mask <str>`: [optional]

  String value. Indicates which mask to apply (possible values: Train, None or All)
  (Default= 'Train')
  
- `--kbest_thresh <int>`: [optional]

  Int value. K value for the K best feature selection.
  (Default= 300)
  
- `--alpha <int>`: [optional]

  Int value (between 0 and 100, multiples of 10). Weight of model 1 in prediction 
  (and 100-alpha is weight of model 2)
  (Default= 70)
  
- `--beta <int>`: [optional]

  Int value (between 10 and 90, multiples of 10). Threshold for prediction brown.
  (Default= 10)
  
- `--10fold`: [optional]

  If in the command, does the 10fold cross validation. If not, does simple cross validation.
  
- `--recompute`: [optional]

  If in the command, does not use the previously computed pickles, recomputes everything.


### Feature selection

An example of features selected are shown in the file `feature_extracted.txt`.
The 300 selected features are ordered by alphabetic order.


© [2020] Ubisoft Entertainment. All Rights Reserved
