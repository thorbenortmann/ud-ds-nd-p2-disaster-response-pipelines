### Table of Contents

1. [Installation](#installation)
2. [Project Motivation and File Descriptions](#project-motivation-and-file-descriptions)
3. [Licensing, Authors, and Acknowledgements](#licensing-authors-acknowledgements)

## Installation
(The commands below always assume you are in the project's root directory.)

### Repo Setup
The code should run with no issues using python versions >=3.6.
1. Create a virtual environment:  
`python -m venv venv`
2. Activate it:  
`venv\Scripts\activate.bat`(for Windows) or `source venv/bin/activate` (for Unix and MacOS)
3. Install all required packages listed in the `requirements.txt` file:  
`pip install -r requirements.txt`

### Create Database
To run the ETL pipeline that takes and cleans data from
`src/data/disaster_messages.csv`and `src/data/disaster_categories.csv` and stores it in a sqlite database
 run:  
`cd src/data`  
`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`  

### Create Model
To run the ML pipeline that trains and saves a classifier based on the created database run:  
`cd src/models`  
`python train_classifier.py ../data/DisasterResponse.db classifier.pkl`  

### Run App
To run the flask web app run:  
`cd src/app`  
`python run.py`  
and go to:
`http://0.0.0.0:3001/` in your browser.

## Project Motivation and File Descriptions
For this project I was interested in classifying messages
that were sent during natural disasters via social media or
directly to disaster response organizations.

### Data
You can find the used data in
`src/data/disaster_messages.csv`
and
`src/data/disaster_categories.csv`.

### ETL Pipeline
To make the data usable to train a classifier I defined an ETL pipeline in
`src/data/process_data.py`.

### Model Training
The model training based on the processed data is defined in
`src/models/train_classifier.py`.

### Visualization and Classification
The data is visualized in a flask web app.
The web app is defined in in the `src/app` folder and can be started from
`src/app/run.py`.
Besides data visualization, you can also enter message texts
and get the classification results of the trained classifier model back.

##Results
The best combination of transformers and classifiers I could find is a
`TfidfVectorizer` using unigrams and bigrams
followed by a (multioutput) `RandomForestClassifier` with 100 trees.

I also experimented with other parameters,
a Naive Bayes classifier (`MultinomialNB`)
and custom transformers.

If you want to edit the classification pipeline you can start by
changing its parameters or even change its components inside the
`build_model()` function in `src/models/train_classifier.py`.
 


## Licensing, Authors, Acknowledgements
Must give credit to
[Figure Eight](https://www.figure-eight.com/)
for the data.
Feel free to use the code here as you would like!