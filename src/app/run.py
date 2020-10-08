import json
from pathlib import Path
import sys
from typing import List

from flask import Flask
from flask import render_template, request
import joblib
import pandas as pd
from pandas import DataFrame, Series
import plotly
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# Make import of other python modules of this project available
project_root: Path = Path(__file__).absolute().parents[2]
sys.path.append(str(project_root))

from src.app import paths
# needed to load the model
from src.models.train_classifier import tokenize


app = Flask(__name__)

# load data
engine = create_engine(f'sqlite:///{str(paths.db_path)}')
df: DataFrame = pd.read_sql_table(paths.table_name, engine)
categories: List[str] = df.columns[4:]

# load model
model = joblib.load(paths.model_path)


# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals

    # example message and categories
    sample: DataFrame = df.sample()
    sample_message: str = sample['message'].values[0]
    sample_categories: Series = sample[categories].iloc[0, :]
    sample_matching_categories: List[str] = sample_categories[sample_categories == 1].index.tolist()

    # Top n categories
    top_n: int = 10
    samples_per_category: Series = df[categories].sum(axis=0).sort_values(ascending=False)
    top_n_categories: Series = samples_per_category.head(top_n)

    # Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        # Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=top_n_categories.index.tolist(),
                    y=top_n_categories.values.tolist()
                )
            ],

            'layout': {
                'title': f'Number of Messages per Category '
                         f'(Top {top_n} out of {len(categories)} Categories)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Number of Message per Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html',
                           ids=ids,
                           graphJSON=graphJSON,
                           message=sample_message,
                           categories=sample_matching_categories)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(categories, classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
