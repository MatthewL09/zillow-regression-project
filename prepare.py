import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import acquire
from env import username, password, host

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

    df = acquire.get_zillow_data()
    df = df.rename( columns = {'bedroomcnt': 'bedroom',
                           'bathroomcnt': 'bathroom',
                           'calculatedfinishedsquarefeet':'square_ft',
                           'lotsizesquarefeet': 'lot_size',
                           'taxvaluedollarcnt': 'tax_value',
                           'yearbuilt': 'year_built',
                           'fips': 'county'
                           })


    # clean all values and replace any missing
    df= df.replace(r'^\s*$', np.nan, regex = True)

    # drop all null values from df 
    df = df.dropna()

    # create column that shows age values calulated from the yearbuilt column
    df['age'] = 2017 - df.year_built

    # replace fips values with locations
    df.county = df.county.replace({6037: 'los_angeles', 6059: 'orange', 6111: 'ventura'})

    # change type for strings
   
    
    # creat dummy columns for fips so it will be easier to evaluate later
    dummy_df = pd.get_dummies(df['county'])
    df = pd.concat([df, dummy_df], axis = 1)


    # remove outliers to something more reasonable
    df =remove_outliers(df, 3.0, ['bedroom', 'bathroom', 'square_ft', 'lot_size', 'tax_value', 'age'])
    # change dtypes for certain columns
    int_col = ['bedroom', 'bathroom', 'square_ft', 'tax_value', 'age']
    for col in df:
        if col in int_col:
            df[col] = df[col].astype(int)

    return df

def scale_data(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits. using the MinMaxScaler()
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['bedroom', 'bathroom', 'square_ft', 'lot_size', 'age']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def split_data(df):
    ''' This function will take in the data and split it into train, validate, and test datasets for modeling, evaluating, and testing
    '''
    train_val, test = train_test_split(df, train_size = .8, random_state = 123)

    train, validate = train_test_split(train_val, train_size = .7, random_state = 123)

    return train, validate, test

def create_x_y(train, validate, test, train_scaled, validate_scaled, test_scaled):
    ''' create X and y versions of split data and
        X and y of scaled data
    '''
    X_train = train_scaled.drop(columns=['tax_value','county','los_angeles','orange','ventura'])
    y_train = pd.DataFrame(train.tax_value)

    X_validate = validate_scaled.drop(columns=['tax_value','county','los_angeles','orange','ventura'])
    y_validate = pd.DataFrame(validate.tax_value)

    X_test = test_scaled.drop(columns=['tax_value','county','los_angeles','orange','ventura'])
    y_test = pd.DataFrame(test.tax_value)

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def plot_residuals(actual, predicted):
    plt.figure(figsize = (10,8))
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':', color ='red')
    sns.scatterplot(actual, residuals, marker = '+')
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()

def wrangle_zillow():
    ''' This function combines both functions above and outputs three cleaned and prepped datasets
    '''
    clean_df = clean_zillow_data()

    train, validate, test = split_data(clean_df)

    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, return_scaler=False)

    return train, validate, test, train_scaled, validate_scaled, test_scaled