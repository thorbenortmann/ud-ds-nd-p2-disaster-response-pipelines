import sys

import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.concat(
        [messages.set_index('id'),
         categories.set_index('id')],
        axis=1,
        join='inner').reset_index()


def clean_data(df: DataFrame) -> DataFrame:
    categories = df['categories'].str.split(';', expand=True)
    category_col_names = [col_name[:-2] for col_name in categories.iloc[0].values]
    categories.columns = category_col_names

    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1, join='inner')

    df = df.drop_duplicates()
    df = df.drop(df[df['related'] == 2].index)

    return df


def save_data(df: DataFrame, database_filename: str) -> None:
    engine = create_engine(f'sqlite:///{database_filename}')
    db_name = database_filename.split("/")[-1]
    table_name = db_name[:-3]
    df.to_sql(table_name, engine, index=False)
    print(f'Saved data to table {table_name} in database {db_name}')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        print(f'and DATABASE TABLE {database_filepath[:-2]}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv ''DisasterResponse.db')


if __name__ == '__main__':
    main()
