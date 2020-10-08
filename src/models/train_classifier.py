"""
Module/script that trains a classifier
based on data in a given database
and stores it at a given path.
"""

import sys
from typing import List, Tuple

import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def load_data(database_filepath: str) -> Tuple[Series, DataFrame, List[str]]:
    """
    Loads the data from the given db and returns it as a triple of:
        - source column (X)
        - target columns (Y)
        - target labels
    :param database_filepath: path to the db to load the data from
    :return: source column, target columns, target labels.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    db_name = database_filepath.split("/")[-1]
    table_name = db_name[0:-3]
    df = pd.read_sql_table(table_name, engine)

    X = df['message']
    category_names = df.columns[4:]
    Y = df[category_names]

    return X, Y, category_names


def tokenize(text: str) -> List[str]:
    """
    Processes the given text into tokens (words). Applied steps are:
        - (word) tokenization
        - lemmatization
        - to lower case
        - whitespace removal
        - stopword removal
        - punctuation removal
    :param text: string to be tokenized.
    :return: a list of the created tokens.
    """
    # the following import has to happen here to be able to fit GridSearchCV in parallel.
    from nltk.corpus import stopwords
    tokens = word_tokenize(text)

    # lemmatize, to lower and whitespace removal
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    # stopword and punctuation removal
    stop_words = set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in stop_words and tok.isalpha()]

    return tokens


def build_model() -> GridSearchCV:
    """
    Creates a GridSearchCV object based on a Pipeline, which combines
    vectorization and classification algorithms to classify given text inputs
    into many target classes.
    :return: the described GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': (50, 100)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-2, verbose=1)
    return cv


def evaluate_model(model: GridSearchCV, X_test: Series, Y_test: DataFrame, category_names: List[str]) -> None:
    """
    Evaluates the given model on a (test) data set it was not trained on by computing
    Precision, Recall, F1-Score and Accuracy for each category.
    :param model: model to be evaluated
    :param X_test: test source data to be classified by the model
    :param Y_test: test target data; the expected labels
    :param category_names: names of the target labels
    :return: None; evaluation results are printed.
    """
    Y_pred = model.predict(X_test)
    cp = classification_report(Y_test.values, Y_pred, target_names=category_names)
    print("Classification Report:\n", cp)

    accuracy = (Y_pred == Y_test).mean()
    print("\naverage Accuracy:\n", accuracy.mean())


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Saves the best estimator of the given model to the given file path using joblib.
    :param model: GridSearchCV object that contains the best estimator to store
    :param model_filepath: path to store the best estimator to
    :return: None
    """
    print(f'Found the following parameters to be the best:\n{model.best_params_}')
    joblib.dump(model.best_estimator_, model_filepath)


def main() -> None:
    """
    Main method of the module/script which uses the given sys.argv to determine the location
    of the input db and the location of the file to store the trained classifier to.
    If any of those two additional arguments are not given, the main method will only print an error message.
    If all arguments are given, the data from the db is used to train a classifier, that is
    then stored at the specified location.
    :return: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
