import pandas as pd
import numpy as np

# FUNCIONES ORIGINALES
'''
def calculate_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of non-missing values, missing values, and the percentage of missing values
    for each column in a DataFrame, and return them as a sorted DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame for which to calculate NA statistics.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns representing:
        - 'datos sin NAs en q': Number of non-missing values for each column
        - 'Na en q': Number of missing values for each column
        - 'Na en %': Percentage of missing values for each column, sorted in descending order.
    """
    qsna = df.shape[0] - df.isnull().sum(axis=0)
    qna = df.isnull().sum(axis=0)
    ppna = np.round(100 * (df.isnull().sum(axis=0) / df.shape[0]), 2)
    aux = {'datos sin NAs en q': qsna, 'Na en q': qna, 'Na en %': ppna}
    na = pd.DataFrame(data=aux)
    return na.sort_values(by='Na en %', ascending=False)

def val_cat_unicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imprime los valores únicos de las columnas categóricas de un DataFrame.

    Parámetros:
        df (pd.DataFrame): El DataFrame que contiene los datos.
    """
    # Filtrar las columnas categóricas
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns

    # Obtener e imprimir los valores únicos de las columnas categóricas
    for col in categorical_cols:
        print(f"Valores únicos en la columna '{col}':")
        print(df[col].unique())
        print()  # Línea en blanco para separar los resultados
'''

#FUNCIONES ADAPTADAS

def calculate_null(df: pd.DataFrame, categorical_cols=None, numerical_cols=None) -> pd.DataFrame:
    """
    Calculate the number of non-missing values, missing values, and the percentage of missing values
    for each column in a DataFrame, and return them as a sorted DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame for which to calculate NA statistics.
    categorical_cols : list, optional
        List of categorical columns to consider. If None, consider all columns.
    numerical_cols : list, optional
        List of numerical columns to consider. If None, consider all columns.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns representing:
        - 'datos sin NAs en q': Number of non-missing values for each column
        - 'Na en q': Number of missing values for each column
        - 'Na en %': Percentage of missing values for each column, sorted in descending order.
    """
    # Si no se pasan las listas de columnas, usar todas las columnas del DataFrame
    if categorical_cols is None and numerical_cols is None:
        selected_columns = df.columns
    else:
        # Si se pasan las columnas categóricas y/o numéricas, combinarlas
        selected_columns = categorical_cols + numerical_cols
    
    # Calcular los valores nulos solo para las columnas seleccionadas
    qsna = df[selected_columns].shape[0] - df[selected_columns].isnull().sum(axis=0)
    qna = df[selected_columns].isnull().sum(axis=0)
    ppna = np.round(100 * (df[selected_columns].isnull().sum(axis=0) / df[selected_columns].shape[0]), 2)
    
    # Crear un DataFrame con los resultados
    aux = {'datos sin NAs en q': qsna, 'Na en q': qna, 'Na en %': ppna}
    na = pd.DataFrame(data=aux)
    
    # Devolver los resultados ordenados por el porcentaje de valores nulos
    return na.sort_values(by='Na en %', ascending=False)


def val_cat_unicos(df: pd.DataFrame, categorical_cols: list = None) -> None:
    """
    Imprime los valores únicos de columnas categóricas en un DataFrame.
    
    Si no se proporciona una lista de columnas categóricas, las detecta automáticamente.

    Parámetros:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        categorical_cols (list, opcional): Lista de nombres de columnas categóricas. 
                                           Si no se proporciona, se detectarán automáticamente.
    """
    # Detectar automáticamente las columnas categóricas si no se proporciona categorical_cols
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

    # Iterar sobre las columnas categóricas y mostrar los valores únicos
    for col in categorical_cols:
        if col in df.columns:
            print(f"Valores únicos en la columna '{col}':")
            print(df[col].unique())
            print()  # Línea en blanco para separar los resultados
        else:
            print(f"La columna '{col}' no existe en el DataFrame.")


# utils.py
import pandas as pd

def val_num_unicos(df: pd.DataFrame, numeric_cols: list = None) -> None:
    """
    Imprime los valores únicos de columnas numéricas en un DataFrame.
    
    Si no se proporciona una lista de columnas numéricas, las detecta automáticamente.

    Parámetros:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        numeric_cols (list, opcional): Lista de nombres de columnas numéricas. 
                                       Si no se proporciona, se detectarán automáticamente.
    """
    # Detectar automáticamente las columnas numéricas si no se proporciona numeric_cols
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Iterar sobre las columnas numéricas y mostrar los valores únicos
    for col in numeric_cols:
        if col in df.columns:
            print(f"Valores únicos en la columna '{col}':")
            print(df[col].unique())
            print()  # Línea en blanco para separar los resultados
        else:
            print(f"La columna '{col}' no existe en el DataFrame.")


# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Return True for outliers
    return (data < lower_bound) | (data > upper_bound)


