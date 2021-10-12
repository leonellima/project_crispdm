import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def count_subitems(arr, separator=","):
    '''
    INPUT:
    arr - string array
    separator - separator to split string
    
    OUTPUT:
    Dictionary with all the substrings found, and each one with its respective match counter in the entire array
    '''
    items = defaultdict(int)
    for string in arr:
        substrings_list = string.split(separator)
        for substring in substrings_list:
            if substring:
                if substring in items:
                    items[substring] += 1
                else:
                    items[substring] = 0
    return items


def items_statistics(df, column_property, column_for_statistics, items_list):
    items = dict.fromkeys(items_list, 0)
    squares = dict.fromkeys(items_list, 0)
    denoms = dict.fromkeys(items_list, 0)
    
    for item in items_list:
        item_for_search = item.replace("(", "\(")
        item_for_search = item_for_search.replace(")", "\)")
        df_rows_with_item = df[df[column_property].str.contains(item_for_search)]
        items[item] = df_rows_with_item[column_for_statistics].sum().sum()
        squares[item] = (df_rows_with_item[column_for_statistics] ** 2).sum().sum()
        denoms[item] = df_rows_with_item.shape[0]

    df_items = pd.DataFrame(pd.Series(items)).reset_index()
    df_squares = pd.DataFrame(pd.Series(squares)).reset_index()
    df_denoms = pd.DataFrame(pd.Series(denoms)).reset_index()
    
    # Change the column names
    df_items.columns = [column_property, 'sum']
    df_squares.columns = [column_property, 'squares']
    df_denoms.columns = [column_property, 'total']
    
    # Merge dataframes
    df_means = pd.merge(df_items, df_denoms)
    df_all = pd.merge(df_means, df_squares)
    
    # Additional columns needed for analysis
    df_all['mean'] = df_means['sum'] / df_means['total']
    df_all['var'] = df_all['squares'] / df_all['total'] - df_all['mean'] ** 2
    df_all['std'] = np.sqrt(df_all['var'])
    df_all['lower_95'] = df_all['mean'] - 1.96*df_all['std'] / np.sqrt(df_all['total'])
    df_all['upper_95'] = df_all['mean'] + 1.96*df_all['std'] / np.sqrt(df_all['total'])
    
    return df_all
    
    
def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default True, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.figure(figsize=(15, 8))
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test
    
    
    
    
    
    