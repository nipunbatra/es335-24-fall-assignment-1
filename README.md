# Assignment 1

## Total 20 marks (Will be scaled down to 10 marks)

# Human Activity Recognition (HAR)
Human Activity Recognition (HAR) refers to the capability of machines to identify various activities performed by the users. The knowledge acquired from these systems/algorithms is integrated into many applications where the associated device uses it to identify actions or gestures and performs predefined tasks in response.

## Dataset
We are interested in classifying human activities based on accelerometer data. we will be using a publically available dataset called [UCI-HAR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8567275). The dataset is available to download [here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Just for your reference a youtube video of the authors collecting participant's accelerometer data is also available [here](http://www.youtube.com/watch?v=XOEN9W05_4A). 

## Task 1 : Exploratory Data Analysis (EDA) [3 marks]

### Preprocessing
We will use the raw accelerometer data within the inertial_signals folder. The provided script, `CombineScript.py`, organizes and sorts accelerometer data, establishing separate classes for each category and compiling participant data into these classes. `MakeDataset.py` script is used to read through all the participant data and create a single dataset. The dataset is then split into train,test and validation set. We focus on the first 10 seconds of activity, translating to the initial 500 data samples due to a sampling rate of 50Hz.

* **Step-1>** Place the `CombineScript.py` and `MakeDataset.py` in the same folder that contains the UCI dataset. Ensure you have moved into the folder before running the scripts. If you are runing the scripts from a different folder, you will have to play around with the paths in the scripts to make it work.
* **Step-2>** Run `CombineScript.py` and provide the paths to test and train folders in UCI dataset. This will create a folder called `Combined` which will contain all the data from all the participants. This is how most of the datasets are organized. You may encounter similar dataset structures in the future.
* **Step-3>** Run `MakeDataset.py` and provide the path to `Combined` folder. This will create a Dataset which will contain the train, test and validation set. You can use this dataset to train your models.


### Questions

1. Plot the waveform for one sample data from each activity class. Are you able to see any difference/similarities between the activities? You can plot a subplot having 6 columns to show differences/similarities between the activities. Do you think the model will be able to classify the activities based on the data? **[0.5 marks]**
2. Do you think we need a machine learning model to differentiate between static activities (laying, sitting, standing) and dynamic activities(walking, walking_downstairs, walking_upstairs)? Look at the linear acceleration $(acc_x^2+acc_y^2+acc_z^2)$ for each activity and justify your answer. **[0.5 marks]**
3. Visualize the data using PCA. **[1 marks]**
    * Use PCA (Principal Component Analysis) on Total Acceleration $(acc_x^2+acc_y^2+acc_z^2)$ to compress the acceleration timeseries into two features and plot a scatter plot to visualize different class of activities. 
    *  Next, use [TSFEL](https://tsfel.readthedocs.io/en/latest/) ([a featurizer library](https://github.com/fraunhoferportugal/tsfel)) to create features (your choice which ones you feel are useful) and then perform PCA to obtain two features. Plot a scatter plot to visualize different class of activities. 
    *  Now use the features provided by the dataset and perform PCA to obtain two features. Plot a scatter plot to visualize different class of activities.
    *  Compare the results of PCA on Total Acceleration, TSFEL and the dataset features. Which method do you think is better for visualizing the data? 
4. Calculate the correlation matrix of the features obtained by TSFEL and provided in the dataset. Identify the features that are highly correlated with each other. Are there any redundant features? **[1 marks]**


## Task 2 : Decision Trees for Human Activity Recognition [3 marks]

### Questions

1. Use Sklearn Library to train Decision Tress. **[1.5 marks]**
    * Train a decision tree model using the raw accelerometer data. Report the accuracy, precision, recall and confusion matrix of the model. 
    * Train a decision tree model using the features obtained by TSFEL. Report the accuracy, precision, recall and confusion matrix of the model. 
    * Train a decision tree model using the features provided in the dataset. Report the accuracy, precision, recall and confusion matrix of the model. 
    * Compare the results of the three models. Which model do you think is better? 
2. Train Decision Tree with varying depths (2-8) using all above 3 methods. Plot the accuracy of the model on test data vs the depth of the tree. **[1 marks]**
3. Are there any participants/ activitivies where the Model performace is bad? If Yes, Why? **[0.5 mark]**

## Task 3 : Prompt Engineering for Large Language Models (LLMs) [4 marks]

### Zero-shot and Few Shot Prompting :
Zero-shot prompting involves providing a language model with a prompt or a set of instructions that allows it to generate text or perform a task without any explicit training data or labeled examples. The model is expected to generate high-quality text or perform the task accurately based solely on the prompt and its internal knowledge.

Few-shot prompting is similar to zero-shot prompting, but it involves providing the model with a limited number of labeled examples or prompts that are relevant to the specific task or dataset. The model is then expected to generate high-quality text or perform the task accurately based on the few labeled examples and its internal knowledge.

### Task Description :
You have been provided with a [Python notebook](./HAR/ZeroShot_FewShot.ipynb) that demonstrates how to use zero-shot and few-shot prompting with a language model (LLM). The example in the notebook involves text-based tasks, but LLMs can also be applied to a wide range of tasks (Students intrested in learning more can read [here](https://deepai.org/publication/large-language-models-are-few-shot-health-learners) and [here](https://arxiv.org/pdf/2305.15525v1)). 

Queries will be provided in the form of featurized accelerometer data and the model should predict the activity performed. 
* **Zero shot learning** : The model should be able to predict the activity based on the accelerometer data without any explicit training data or labeled examples. 
* **Few Shot Learning** :The model should also be able to predict the activity based on a limited number of labeled examples or prompts that are relevant to the specific task. 

### Questions

1. Demonstrate how to use Zero-Shot Learning and Few-Shot Learning to classify human activities based on the featurized accelerometer data. Qualitatively demonstrate the performance of Few-Shot Learning with Zero-Shot Learning. Which method performs better? Why?  **[1 marks]**
2. Quantitatively compare the accuracy of Few-Shot Learning with Decision Trees (You may use a subset of the test set if you encounter rate-limiting issues). Which method performs better? Why? **[1 marks]**
3. What are the limitations of Zero-Shot Learning and Few-Shot Learning in the context of classifying human activities based on featurized accelerometer data? **[1 marks]**
4. What does the model classify when given input from an entirely new activity that it hasn't seen before? **[0.5 mark]**
5. Test the model with random data (ensuring the data has the same dimensions and range as the previous input) and report the results. **[0.5 mark]**


## Task 4 : Data Collection in the Wild [4 marks]

## Task Description
For this exercise marks will not depend on what numbers you get but on the process you followed Utilize apps like `Physics Toolbox Suite` from your smartphone to collect your data in .csv/.txt format. Ensure at least 15 seconds of data is collected, trimming edges to obtain 10 seconds of relevant data. Also record a video of yourself while recording data. This video will be required in some future assignments. Collect 3-5 samples per activity class.

### Things to take care of:
* Ensure the phone is placed in the same position for all the activities.
* Ensure the phone is in the same alignment during the activity as changing the alignment will change the data collected and will affect the model's performance.
* Ensure to have atleast 10s of data per file for training. As the data is collected at 50Hz, you will have 500 data samples.

### Questions
1. Use the Decision Tree model trained on the UCI-HAR dataset to predict the activities that you performed. Report the accuracy, precision, recall and confusion matrix of the model. You have three version of UCI dataset you can use a)Raw data from accelerometer, b)TSFEL featurised data, c)Features provided by author. Choose which version to use, ensuring that your test data is similar to your training data. How did the model perform? **[1 marks]**
2. Use the data you collected to predict the activities that you performed. Decide whether to apply preprocessing and featurization, and if so, choose the appropriate methods. How did the model perform? **[1 marks]**
3. Use the Few-Shot prompting method using UCI-HAR dataset to predict the activities that you performed. Ensure that both your examples and test query undergo similar preprocessing. How did the model perform? **[1 marks]**
4. Use the Few-Shot prompting method using the data you collected to predict the activities that you performed. Adopt proper processing methods as needed. How did the model perform? **[1 marks]**

#### NOTE :
1. To obtain API key go to the GroqCloud Developer Console at https://console.groq.com/login. Follow the Quickstart guide to obtain your API key.
2. ***DO NOT share your API key with anyone or make it public or upload it to any public repository such as for this assignment. If the key is found in the code, you will be penalized with a <u>1.0 marks deduction</u>.***
3. It is advised to either write a markdown file (.md) or use a Python notebook (.ipynb) to demonstrate your reasoning, results and findings.

## Decision Tree Implementation [6 marks]

1. Complete the decision tree implementation in tree/base.py. The code should be written in Python and not use existing libraries other than the ones shared in class or already imported in the code. Your decision tree should work for four cases: i) discrete features, discrete output; ii) discrete features, real output; iii) real features, discrete output; real features, real output. <u>Your model should accept real inputs only (for discrete inputs, you may convert the attributes into one-hot encoded vectors)</u>. Your decision tree should be able to use InformationGain using Entropy or GiniIndex as the criteria for splitting for discrete output. Your decision tree should be able to use InformationGain using MSE as the criteria for splitting for real output. Your code should also be able to plot/display the decision tree.  **[2.5 marks]**

    > You should be editing the following files.
  
    - `metrics.py`: Complete the performance metrics functions in this file. 

    - `usage.py`: Run this file to check your solutions.

    - tree (Directory): Module for decision tree.
      - `base.py` : Complete Decision Tree Class.
      - `utils.py`: Complete all utility functions.
      - `__init__.py`: **Do not edit this**

    > You should run _usage.py_ to check your solutions. 

1. 
    Generate your dataset using the following lines of code

    ```python
    from sklearn.datasets import make_classification
    X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

    # For plotting
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    ```

    a) Show the usage of *your decision tree* on the above dataset. The first 70% of the data should be used for training purposes and the remaining 30% for test purposes. Show the accuracy, per-class precision and recall of the decision tree you implemented on the test dataset. **[0.5 mark]**

    b) Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree. **[1 mark]**
    
    > You should be editing `classification-exp.py` for the code containing the above experiments.

2. 
    a) Show the usage of your decision tree for the [automotive efficiency](https://archive.ics.uci.edu/ml/datasets/auto+mpg) problem. **[0.5 marks]**
    
    b) Compare the performance of your model with the decision tree module from scikit learn. **[0.5 marks]**
    
   > You should be editing `auto-efficiency.py` for the code containing the above experiments.
    
3. Create some fake data to do some experiments on the runtime complexity of your decision tree algorithm. Create a dataset with N samples and M binary features. Vary M and N to plot the time taken for: 1) learning the tree, 2) predicting for test data. How do these results compare with theoretical time complexity for decision tree creation and prediction. You should do the comparison for all the four cases of decision trees. **[1 marks]**	

    >You should be editing `experiments.py` for the code containing the above experiments.


You must answer the subjectve questions (visualization,timing analysis, displaying plots) by creating `Asst#<task-name>_<Q#>.md`


### **Genral Instructions :**
1. Show your results in a Jupyter Notebook or an MD file. If you opt for using an MD file, you should also include the code.
2. You can use the scikit-learn implementation of Decision Tree for the Human Activity Recognition.
3. This assignment is of 20 marks and will be scaled down to 10 marks.

