import pandas as pd
import numpy as np
import os
from typing import Callable, Dict, Any

"""
Purpose: keep all useful pandas dataframe manipulations in one place.
Note: many of these are gpt-generated--LLMs are quite useful for boilerplate
functionalities like these!
"""

# Function 1: Select the first m rows and the first n columns
def select_rows_and_columns(df, m, n):
    """
    Select the first m rows and the first n columns from a dataframe.
    """
    return df.iloc[:m, :n]

# Function 2: Add the p+1st row of largedf to df
# Revised Function 2: Add the p+1st row of largedf with the first q entries
def add_row_from_largedf(df, largedf):
    """
    Add the p+1st row of largedf with the first q entries to df.
    """
    # Size of df
    p, q = df.shape
    # Get the p+1st row (index p since pandas is 0-based) and the first q entries
    row_to_add = largedf.iloc[p, :q]
    # Append the row as a new dataframe
    return pd.concat([df, row_to_add.to_frame().T], ignore_index=True)

# Revised Function 3: Add the first p values of the q+1st column of largedf to df
def add_column_from_largedf(df, largedf):
    """
    Add the first p values of the q+1st column of largedf to df, retaining the column's name.
    """
    # Size of df
    p, q = df.shape
    # Get the first p values of the q+1st column (index q since pandas is 0-based)
    column_name = largedf.columns[q]  # Get the name of the q+1st column
    column_to_add = largedf.iloc[:p, q]
    # Add the column to df with the same name
    df[column_name] = column_to_add.values
    return df

def test_select_rows_and_columns():
    # Example dataframes
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    largedf = pd.DataFrame({
        'A': [7, 8, 9, 10],
        'B': [11, 12, 13, 14],
        'C': [15, 16, 17, 18]
    })
    
    # Test Function 1
    print(select_rows_and_columns(df, 2, 1))

def test_add_row_from_largedf():
    # Example dataframes
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    largedf = pd.DataFrame({
        'A': [7, 8, 9, 10],
        'B': [11, 12, 13, 14],
        'C': [15, 16, 17, 18]
    })
    # Test Function 2
    print(add_row_from_largedf(df, largedf))

def test_add_column_from_largedf():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    largedf = pd.DataFrame({
        'A': [7, 8, 9, 10],
        'B': [11, 12, 13, 14],
        'C': [15, 16, 17, 18]
    })
    # Test Function 3
    print(add_column_from_largedf(df, largedf))



# gpt-4o generated
def filter_and_select(df, c_i, r_vals, c_vals):
    """
    Filters rows where the column `c_i` contains values in `r_vals`
    and selects the columns specified in `c_vals`.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        c_i (str): The column name to filter on.
        r_vals (list): The list of values to filter by in column `c_i`.
        c_vals (list): The list of column names to select from the filtered DataFrame.

    Returns:
        pd.DataFrame: A filtered and selected DataFrame.
    """
    # Filter rows where the column `c_i` contains values in `r_vals`
    filtered_df = df[df[c_i].isin(r_vals)]

    # Select columns `c_vals`
    result_df = filtered_df[c_vals]

    return result_df

def test_filter_and_select():
    # Sample DataFrame
    data = {
        'A': [1, 2, 3, 4],
        'B': ['apple', 'banana', 'cherry', 'date'],
        'C': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    
    # Define parameters
    c_i = 'A'
    r_vals = [2, 4]
    c_vals = ['B', 'C']
    
    # Apply the function
    filtered_df = filter_and_select(df, c_i, r_vals, c_vals)
    print(filtered_df)

def extract_single(df: pd.DataFrame):
    """
    Validates that the dataframe is 1x1 and returns the single cell's value and row index.
    
    Parameters:
        df (pd.DataFrame): The dataframe to validate and extract from.
    
    Returns:
        tuple: A tuple containing the row index and cell value.
    
    Raises:
        ValueError: If the dataframe is not 1x1.
    """
    # Check if the dataframe is 1x1
    if df.shape != (1, 1):
        raise ValueError(f"DataFrame must be 1x1, but got shape {df.shape}.\n\nDataFrame: {df}")
    
    # Extract the single row index and cell value
    row_index = df.index[0]
    cell_value = df.iloc[0, 0]
    
    return row_index, cell_value

def test_validate_and_extract():
    # Example DataFrame
    df = pd.DataFrame({'A': [42]}, index=[5])  # A 1x1 DataFrame with row index 5
    
    # Call the function
    row_index, cell_value = extract_single(df)
    print(f"Row Index: {row_index}, Cell Value: {cell_value}")

def modify_dataframe(df, range1, range2, range3, num_els4, m_f, c_i):
    """
    Modify a DataFrame by applying a function m_f to each cell, 
    except for cells in the column c_i.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        range1 (int): Range for the first random value.
        range2 (int): Range for the second random value.
        range3 (int): Range for the third random value.
        m_f (function): A function that takes three integers and returns a value.
        c_i (str): Column to ignore.

    Returns:
        pd.DataFrame: A new DataFrame with modified values.
    """
    # Create a copy of the dataframe to avoid modifying the original
    new_df = df.copy()
    
    # Iterate over each column
    for col in new_df.columns:
        if col != c_i:
            # Iterate over each cell in the column
            for index in new_df.index:
                # Generate random values in the given ranges
                x1 = np.random.randint(0, range1 + 1)
                x2 = np.random.randint(0, range2 + 1)
                lst3 = list(range(range3))
                x3_arr = np.random.choice(lst3, num_els4, replace=False)
                x3 = x3_arr.tolist()
                
                # Apply the modification function to the random values
                new_value = m_f(x1, x2, x3)
                
                # Update the cell with the new value
                new_df.at[index, col] = new_value
    
    return new_df

def test_modify_dataframe():
    # Example DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'Ignore': [7, 8, 9]
    })
    
    # Example modification function
    def my_func(x1, x2, x3):
        return x1 + x2 * (x3[0] + x3[1])
    
    # Modify the DataFrame
    new_df = modify_dataframe(df, 10, 20, 30, 2, my_func, 'Ignore')
    
    print(new_df)

def generic_modify(df, m_f, c_i):
    """
    Modify a DataFrame by applying a function m_f to each cell, 
    except for cells in the column c_i.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        m_f (function): A function to be applied on each cell value.
        c_i (str): Column to ignore.

    Returns:
        pd.DataFrame: A new DataFrame with modified values.
    """
    # Create a copy of the dataframe to avoid modifying the original
    new_df = df.copy()
    
    # Iterate over each column
    for col in new_df.columns:
        if col != c_i:
            # Iterate over each cell in the column
            for index in new_df.index:
                cur_value = new_df.at[index, col]
                
                # Apply the modification function to the value
                new_value = m_f(cur_value)
                
                # Update the cell with the new value
                new_df.at[index, col] = new_value
    
    return new_df

def verify_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    c_i: str,
    row_func: Callable[[Any, pd.Series], Any],
    col_func: Callable[[str, pd.Index], str],
    cell_func: Callable[[Any, Any], bool]
) -> bool:
    """
    Verify if two dataframes represent the same information based on mapping rules.
    
    Parameters:
    - df1: First DataFrame.
    - df2: Second DataFrame.
    - c_i: Index column name common to both DataFrames.
    - row_func: Function to match rows (row value in df1 to a row value in df2[c_i]).
    - col_func: Function to match columns (column in df1 to a column in df2).
    - cell_func: Function to compare two cell values.
    
    Returns:
    - bool: True if all matched cells are equivalent according to cell_func.
    """
    # Step 1: Build row_map
    row_map = {}
    for val1 in df1[c_i]:
        match = row_func(val1, df2[c_i])
        if match in df2[c_i].values:
            row_map[val1] = match
    
    # Step 2: Build col_map
    col_map = {}
    for col1 in df1.columns:
        match = col_func(col1, df2.columns)
        if match in df2.columns:
            col_map[col1] = match
    
    # Step 3: Verify cell values
    if not row_map or not col_map:
        return False  # If no matches found, verification fails
    
    for row1, row2 in row_map.items():
        for col1, col2 in col_map.items():
            val1 = df1.loc[df1[c_i] == row1, col1].iloc[0]
            val2 = df2.loc[df2[c_i] == row2, col2].iloc[0]
            if not cell_func(val1, val2):
                return False
    
    return True

# Example usage
def example_row_func(val1, series):
    return val1 if val1 in series.values else None

def example_col_func(col1, columns):
    return col1 if col1 in columns else None

def example_cell_func(val1, val2):
    return val1 == val2

def test_verify_df():
    # Example DataFrames
    df1 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob'], 'score': [85, 90]})
    df2 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob'], 'score': [85, 90]})
    
    result = verify_dataframes(df1, df2, 'id', example_row_func, example_col_func, example_cell_func)
    print("DataFrames match:", result)

def count_same_cells(df1, df2, index_col):
    if not df1.columns.equals( df2.columns):
        raise Exception(f"Columns Differ-df1: {df1.columns},\n\ndf2: {df2.columns}")
    elif not df1[index_col].equals(df2[index_col]):
        raise Exception(f"Index column values differ: df1: {df1[index_col]},\n\ndf2: {df2[index_col]}")
    
    num_rows = df1.shape[0]
    num_cols = df1.shape[1]
    
    same_cell_cnt = 0
    diff_cell_cnt = 0
    
    for i in range(num_rows):
        for j in range(num_cols):
            if df1.loc[i][j] == df2.loc[i][j]:
                same_cell_cnt += 1
            else:
                diff_cell_cnt += 1
    
    print(f"Dataframe Shape: {num_rows}, {num_cols}")
    print(f"Matching cells: {same_cell_cnt}")
    print(f"Differing cells: {diff_cell_cnt}")

def test_count_same_cells():
    path1 = os.path.expanduser('~/sociome_data/sociome_revised_sql.csv')
    path2 = os.path.expanduser('~/sociome_data/sociome_revised_badsql.csv')
    testdf1 = pd.read_csv(path1)
    testdf2 = pd.read_csv(path2)
    
    count_same_cells(testdf1, testdf2, 'Role')

def replace_from_df(df, repdf, index_col):
    """
    Replace all cells in df with those from repdf and return. If repdf is smaller than df, then wrap the values around in df
    and return. That is, if we need a new row, then wrap back to first row, and if we need a new column, then wrap back to first non-index column.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    repdf : pd.DataFrame
        The dataframe to replace the input dataframe cells with.
    index_col : str
        The index column of both df and repdf.

    Returns
    -------
    The altered df, a pandas dataframe.
    """
    
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    max_rows = repdf.shape[0]
    max_cols = repdf.shape[1]
    
    for i in range(num_rows):
        for j in range(num_cols):
            if df.columns[j] == 'Role':
                continue
            if i < max_rows and j < max_cols:
                df.loc[i][j] = repdf.loc[i][j]
            elif i >= max_rows and j >= max_cols:
                valid_i = i % max_rows
                valid_j = j % max_cols
                df.loc[i][j] = repdf.loc[valid_i][valid_j]
            elif i >= max_rows:
                valid_i = i % max_rows
                df.loc[i][j] = repdf.loc[valid_i][j]
            elif j >= max_cols:
                valid_j = j % max_cols
                df.loc[i][j] = repdf.loc[i][valid_j]
    
    return df

def test_replace_from_df():
    path1 = os.path.expanduser('~/sociome_data/sociome_revised_sql.csv')
    path2 = os.path.expanduser('~/sociome_data/sociome_revised_badsql.csv')
    path3 = os.path.expanduser('~/sociome_data/sociome_revised_nl.csv')
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    
    rep1 = os.path.expanduser('~/automatedgov/bird_simpletime_european_football_2_sql/bird_simpletime_sql_chunk_11.csv')
    rep2 = os.path.expanduser('~/automatedgov/bird_simpletime_european_football_2_badsql/bird_simpletime_badsql_chunk_11.csv')
    rep3 = os.path.expanduser('~/automatedgov/bird_simpletime_european_football_2_nl/bird_simpletime_nl_chunk_11.csv')
    repdf1 = pd.read_csv(rep1)
    repdf2 = pd.read_csv(rep2)
    repdf3 = pd.read_csv(rep3)
    
    newpath1 = os.path.expanduser('~/sociome_data/sociome_temp_sql.csv')
    newpath2 = os.path.expanduser('~/sociome_data/sociome_temp_badsql.csv')
    newpath3 = os.path.expanduser('~/sociome_data/sociome_temp_nl.csv')
    
    
    new_df1 = replace_from_df(df1, repdf1, 'Role')
    new_df2 = replace_from_df(df2, repdf2, 'Role')
    new_df3 = replace_from_df(df3, repdf3, 'Role')
    
    new_df1.to_csv(newpath1, index=False)
    new_df2.to_csv(newpath2, index=False)
    new_df3.to_csv(newpath3, index=False)
    
def replace_df_column(df, newdf, rep_col):
    if rep_col not in df.columns:
        raise Exception(f"Column {rep_col} not in df: {df.columns}")
    elif rep_col not in newdf.columns:
        raise Exception(f"Column {rep_col} not in newdf: {newdf.columns}")
    
    num_rows = df.shape[0]
    new_num_rows = newdf.shape[0]
    
    if new_num_rows < num_rows:
        outdf = df.head(new_num_rows)
        outdf[rep_col] = newdf[rep_col]
    elif new_num_rows == num_rows:
        outdf = df.copy()
        outdf[rep_col] = newdf[rep_col]
    elif new_num_rows > num_rows:
        rep_head = df.head(new_num_rows - num_rows)
        outdf = pd.concat([df, rep_head], ignore_index=True)
        outdf[rep_col] = newdf[rep_col]
    
    return outdf

def read_then_replace(path1, path2, rep_col, new_name):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    newdf = replace_df_column(df1, df2, rep_col)
    
    newdf.to_csv(new_name, index=False)

def test_replace_df_column():
    base_nl1 = os.path.expanduser('~/automatedgov/amazon_nlacm_wide_nl/amazon_nlacm_wide_nl_base.csv')
    base_sql1 = os.path.expanduser('~/automatedgov/amazon_nlacm_wide_sql/amazon_nlacm_wide_sql_base.csv')
    
    base_nl2 = os.path.expanduser('~/automatedgov/amazon_nlacm_balance_nl/amazon_nlacm_balance_nl_base.csv')
    base_sql2 = os.path.expanduser('~/automatedgov/amazon_nlacm_balance_sql/amazon_nlacm_balance_sql_base.csv')
    
    tmp1 = os.path.expanduser('~/automatedgov/bird_simpletime_european_football_2_sql/bird_simpletime_sql_chunk_11.csv')
    tmp2 = os.path.expanduser('~/automatedgov/bird_simpletime_european_football_2_badsql/bird_simpletime_badsql_chunk_11.csv')
    tmp3 = os.path.expanduser('~/automatedgov/bird_simpletime_european_football_2_nl/bird_simpletime_nl_chunk_11.csv')
    
    priv1 = os.path.expanduser('~/automatedgov/bird_nlacm_european_football_2_sql/bird_nlacm_sql_chunk_11.csv')
    priv2 = os.path.expanduser('~/automatedgov/bird_nlacm_european_football_2_badsql/bird_nlacm_badsql_chunk_11.csv')
    priv3 = os.path.expanduser('~/automatedgov/bird_nlacm_european_football_2_nl/bird_nlacm_nl_chunk_11.csv')
    
    rep_col = 'Role'
    
    #NL Dataframes first
    read_then_replace(tmp3, base_nl1, rep_col, 'wide_simpletime_nl.csv')
    read_then_replace(priv3, base_nl1, rep_col, 'wide_nlacm_nl.csv')
    
    #then SQL
    read_then_replace(tmp1, base_sql1, rep_col, 'wide_simpletime_sql.csv')
    read_then_replace(priv1, base_sql1, rep_col, 'wide_nlacm_sql.csv')
    
    #then bad SQL
    read_then_replace(tmp2, base_sql1, rep_col, 'wide_simpletime_badsql.csv')
    read_then_replace(priv2, base_sql1, rep_col, 'wide_nlacm_badsql.csv')
    
    #NL Dataframes first
    read_then_replace(tmp3, base_nl2, rep_col, 'balance_simpletime_nl.csv')
    read_then_replace(priv3, base_nl2, rep_col, 'balance_nlacm_nl.csv')
    
    #then SQL
    read_then_replace(tmp1, base_sql2, rep_col, 'balance_simpletime_sql.csv')
    read_then_replace(priv1, base_sql2, rep_col, 'balance_nlacm_sql.csv')
    
    #then bad SQL
    read_then_replace(tmp2, base_sql2, rep_col, 'balance_simpletime_badsql.csv')
    read_then_replace(priv2, base_sql2, rep_col, 'balance_nlacm_badsql.csv')

if __name__=='__main__':
    #use for one-off testing these functions
    # test_validate_and_extract()
    # test_modify_dataframe()
    # test_verify_df()
    # test_select_rows_and_columns()
    # test_add_column_from_largedf()
    # test_add_row_from_largedf()
    # test_count_same_cells()
    # test_replace_from_df()
    test_replace_df_column()


