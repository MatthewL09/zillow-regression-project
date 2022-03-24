import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from env import username, password, host


def get_zillow_data(use_cache=True):
    '''This function returns the data from the zillow database in Codeup Data Science Database. 
    In my SQL query I have joined all necessary tables together, so that the resulting dataframe contains all the 
    information that is needed
    '''
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/'
    query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    JOIN predictions_2017 USING(parcelid)
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential") AND transactiondate LIKE " 2017%%";
    '''
    df = pd.read_sql(query, database_url_base + 'zillow')
    df.to_csv('zillow.csv', index=False)
    return df


def clean_zillow_data():
    '''This function will take in the acquired data and clean it by replacing any white spaces
    dropping any null values and renaming columns for better readability
    '''

    df = get_zillow_data()
    df = df.rename( columns = {'bedroomcnt': 'bedroom',
                           'bathroomcnt': 'bathroom',
                           'calculatedfinishedsquarefeet':'square_ft',
                           'taxvaluedollarcnt': 'tax_value',
                           'yearbuilt': 'year_built',
                          'taxamount':'tax'})

    # clean all values and replace any missing
    df= df.replace(r'^\s*$', np.nan, regex = True)

    # drop all null values from df and confirm
    df = df.dropna()

    # change all dtype to integer from floats
    df = df.astype('int')

    return df

def split_data(df):
    ''' This function will take in the data and split it into train, validate, and test datasets for modeling, evaluating, and testing
    '''
    train_val, test = train_test_split(df, train_size = .8, random_state = 123)

    train, validate = train_test_split(train_val, train_size = .7, random_state = 123)

    return train, validate, test

def wrangle_zillow():
    ''' This function combines both functions above and outputs three cleaned and prepped datasets
    '''
    clean_df = clean_zillow_data()
    train, validate, test = split_data(clean_df)

    return train, validate, test