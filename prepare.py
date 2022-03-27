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
SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, lotsizesquarefeet, taxvaluedollarcnt, yearbuilt, fips
FROM properties_2017
JOIN predictions_2017 USING(parcelid)
LEFT JOIN propertylandusetype USING(propertylandusetypeid)
WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential") AND transactiondate LIKE '2017%%';
    '''
    df = pd.read_sql(query, database_url_base + 'zillow')
    df.to_csv('zillow.csv', index=False)
    return df

### a function that will be needed in the clean_zillow_data to remove outliers
def remove_outliers(df, k, col_list):
    ''' this function take in a dataframe, k value, and specified columns 
    within a dataframe and then return the dataframe with outliers removed
    '''
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def clean_zillow_data():
    '''This function will take in the acquired data and clean it by replacing any white spaces
    dropping any null values and renaming columns for better readability
    '''

    df = get_zillow_data()
    df = df.rename( columns = {'bedroomcnt': 'bedroom',
                           'bathroomcnt': 'bathroom',
                           'calculatedfinishedsquarefeet':'square_ft',
                           'lotsizesquarefeet': 'lot_size',
                           'taxvaluedollarcnt': 'tax_value',
                           'yearbuilt': 'year_built'
                           })

    # clean all values and replace any missing
    df= df.replace(r'^\s*$', np.nan, regex = True)

    # create column that shows age values calulated from the yearbuilt column
    df['age'] = 2022 - df.year_built

    # replace fips values with locations
    df.fips = df.fips.replace({6037: 'los_angeles', 6059: 'orange', 6111: 'ventura'})

    # change type for strings
    df.fips.astype('object')

    # drop all null values from df 
    df = df.dropna()
    
    # creat dummy columns for fips so it will be easier to evaluate later
    dummy_df = pd.get_dummies(df['fips'])
    df = pd.concat([df, dummy_df], axis = 1)

    # remove outliers to something more reasonable
    df =remove_outliers(df, 1.5, ['bedroom', 'bathroom', 'square_ft', 'lot_size', 'tax_value', 'age', 'year_built'])
    # change dtypes for certain columns
    int_col = ['bedroom', 'bathroom', 'square_ft', 'tax_value', 'age']
    for col in df:
        if col in int_col:
            df[col] = df[col].astype(int)

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