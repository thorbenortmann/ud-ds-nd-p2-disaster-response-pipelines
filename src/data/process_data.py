"""
Module/script which processes two csv-files and combines them into one sqlite database.
The paths to the csv-files and the database have to be passed as sys.args when executing the script.
"""

import sys
from typing import List

import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    """
    Loads two csv files into DatFrames and inner joins them using their 'id' column.
    :param messages_filepath: path to the messages file
    :param categories_filepath: path to the categories file
    :return: a DataFrame containing the joined data of the two files.
    """
    messages: DataFrame = pd.read_csv(messages_filepath)
    categories: DataFrame = pd.read_csv(categories_filepath)
    return pd.concat(
        [messages.set_index('id'),
         categories.set_index('id')],
        axis=1,
        join='inner').reset_index()


def clean_data(df: DataFrame) -> DataFrame:
    """
    Cleans the data of the given DataFrame. This includes:
        - converting the data into the 1 NF by splitting the 'categories' column into separate columns.
        - giving the columns appropriate data types
        - dropping duplicates
        - dropping ambiguous values
    :param df: DataFrame to clean
    :return: the cleaned DataFrame.
    """
    categories: DataFrame = df['categories'].str.split(';', expand=True)
    category_col_names: List[str] = [col_name[:-2] for col_name in categories.iloc[0].values]
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
    """
    Saves the given DataFrame into a sqlite database which is created at the given database_filename.
    :param df: DataFrame to store
    :param database_filename: path where the sqlite db will be created
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    db_name: str = database_filename.split("/")[-1]
    table_name: str = db_name[:-3]
    df.to_sql(table_name, engine, index=False)
    print(f'Saved data to table {table_name} in database {db_name}')


def main() -> None:
    """
    Main method of the module/script which uses the given sys.argv to determine the location
    of the two csv input files it needs and the location where a sqlite database shall be created.
    If any of those three additional arguments are not given, the main method will only print an error message.
    If all of those arguments are given, the two input files are loaded, processed and stored into a sqlite db.
    :return: None
    """
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
              'data sets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv ''DisasterResponse.db')


if __name__ == '__main__':
    main()
