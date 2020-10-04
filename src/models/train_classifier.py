import sys
from typing import List, Tuple

import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath: str) -> Tuple[Series, DataFrame, List[str]]:
    engine = create_engine(f'sqlite:///{database_filepath}')
    db_name = database_filepath.split("/")[-1]
    table_name = db_name[0:-3]
    df = pd.read_sql_table(table_name, engine)

    X = df['message'].head(1000)
    category_names = df.columns[4:]
    Y = df[category_names].head(1000)

    return X, Y, category_names


def tokenize(text: str) -> List[str]:
    tokens = word_tokenize(text)

    # lemmatize, to lower and whitespace removal
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    # stopword and punctuation removal
    stop_words = set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in stop_words and tok.isalpha()]

    return tokens


def build_model() -> GridSearchCV:
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': (5, 10)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    return cv


def evaluate_model(model: GridSearchCV, X_test: Series, Y_test: DataFrame, category_names: List[str]) -> None:
    Y_pred = model.predict(X_test)
    cp = classification_report(Y_test.values, Y_pred, target_names=category_names)
    print("Classification Report:\n", cp)


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    print(f'Found the following parameters to be the best:\n{model.best_params_}')
    joblib.dump(model.best_estimator_, model_filepath)


def main():
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
