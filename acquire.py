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