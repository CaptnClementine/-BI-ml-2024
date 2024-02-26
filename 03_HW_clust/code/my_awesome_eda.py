import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore') #only for mad warning


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    '''
    If less than 10% of your column are the same, this function will make them categorical type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with columns converted to categorical type.

    Example:
        df = to_categorical(df)
    '''
    unique_var_procent = df.apply(lambda x: len(x.unique()) / len(x), axis=0)
    for i in unique_var_procent.index:
        if unique_var_procent.loc[i] <= 0.1:
            df[i] = df[i].astype("category")
    return df


def count_categorical_stat(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Count the unique values and frequencies of categorical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing counts and frequencies of unique values in categorical columns.

    Example:
        categorical_stats = count_categorical_stat(df)
    '''
    categorical_stat = pd.DataFrame()
    categorical_stat['counts'] = df[df.select_dtypes(include='category').columns].nunique()
    categorical_stat['frequencies'] = df[df.select_dtypes(include='category').columns].nunique() / len(df)
    return categorical_stat


def count_numerical_stat(df: pd.DataFrame) -> None:
    '''
    Calculate statistics and identify outliers for numerical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None

    Example:
        count_numerical_stat(df)
    '''
    numerical_stat = df[df.select_dtypes(include='number').columns].describe().reindex(['min', 'max', 'mean', 'std', 'median', 'mad', '25%', '50%', '75%']).T
    numerical_stat['median'] = df[df.select_dtypes(include='number').columns].median()
    numerical_stat['mad'] = df[df.select_dtypes(include='number').columns].mad()
    print(f'\n***** Descriptive Statistics for Numerical Columns *****\n{numerical_stat}')
    numerical_stat['iqr'] = (numerical_stat['75%'] - numerical_stat['25%'])
    numerical_stat['lower_bound'] = numerical_stat['25%'] - 1.5 * numerical_stat['iqr']
    numerical_stat['upper_bound'] = numerical_stat['75%'] + 1.5 * numerical_stat['iqr']
    print(f'\n\n***** Main Dataframe Purity Info *****\n')
    for column in df.select_dtypes(include='number').columns:
        print(f'Column {column} has {sum((df[column] < numerical_stat.loc[column]["lower_bound"]) | (df[column] > numerical_stat.loc[column]["upper_bound"]))} outliers')
    return None


def run_eda(df: pd.DataFrame) -> None:
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - None
    """
    print(f'Hi! I will give you some basic summary of your  dataset. \n   Shape: {len(df)} rows x {df.shape[1]} columns \n')
    df = to_categorical(df)
    print(f'\n***** Data Types in Columns *****\n{df.dtypes.to_string()}\n\n')
    print(f'\n\n***** Main Statistics on Categorical Columns ***** \n{count_categorical_stat(df)}')
    count_numerical_stat(df)
    total_na = df.isna().sum().sum()
    rows_with_na = df[df.isna().any(axis=1)].shape[0]
    columns_with_na = df.columns[df.isna().any(axis=0)].tolist()
    duplicate_rows = df.duplicated().sum()

    print(f"\nTotal missing values in df: {total_na}")
    missing_values = df.isnull().mean()
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    missing_plot = sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
    missing_plot.set_xticklabels(missing_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Proportion of missing values for variables')
    plt.show()
    
    
    print(f"Quantity of  rows with missing values: {rows_with_na}")
    print(f"Columns with missing values: {columns_with_na}")
    print(f"Duplicate rows in df: {duplicate_rows}")
    
    correlation_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title('Heatmap')
    plt.show()
    
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    fig, axes = plt.subplots(nrows=2*len(numerical_columns), ncols=1, figsize=(6, 2 * 2 * len(numerical_columns)))
    fig.tight_layout(pad=4.0)

    for i, column in enumerate(numerical_columns):
        j = i*2
        sns.histplot(df[column], kde=False, ax=axes[j+1], bins=20, color="skyblue")
        axes[j+1].set_title(f'Histogram of {column}')
        sns.boxplot(x=df[column], ax=axes[j], color="lightcoral")
        axes[j].set_title(f'Boxplot of {column}')

    plt.show()


    return(None)