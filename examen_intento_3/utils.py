import pandas as pd
import numpy as np

# Normalizar nombres de columnas
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de las columnas de un DataFrame.
    
    Transforma los nombres para:
    - Eliminar espacios iniciales y finales.
    - Reemplazar caracteres no deseados ('-', espacios, apóstrofes) por '_'.
    - Convertirlos a minúsculas.
    
    Parámetros:
    ----------
    df : pd.DataFrame
        DataFrame cuyos nombres de columnas serán normalizados.

    Retorna:
    -------
    pd.DataFrame
        El DataFrame con columnas normalizadas.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )
    return df

# Función para identificar y contar registros duplicados
def identificar_duplicados(df: pd.DataFrame) -> int:
    """
    Identifica los registros duplicados en un DataFrame y cuenta cuántos existen.

    Parámetros:
    ----------
    df : pd.DataFrame
        El DataFrame donde se buscarán los registros duplicados.

    Retorna:
    -------
    int
        El número de registros duplicados en el DataFrame.
    """
    # Identificar duplicados
    duplicados = df.duplicated()
    
    # Contar el número de duplicados
    num_duplicados = duplicados.sum()
    
    print(f"Número de registros duplicados: {num_duplicados}")
    
    return num_duplicados

# Calcular valores nulos
def calculate_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas sobre valores nulos en un DataFrame.

    Parámetros:
    ----------
    df : pd.DataFrame
        DataFrame para calcular estadísticas de valores nulos.

    Retorna:
    -------
    pd.DataFrame
        Un DataFrame con información sobre:
        - Número de valores no nulos.
        - Número de valores nulos.
        - Porcentaje de valores nulos.
    """
    qsna = df.shape[0] - df.isnull().sum(axis=0)
    qna = df.isnull().sum(axis=0)
    ppna = np.round(100 * (df.isnull().sum(axis=0) / df.shape[0]), 2)
    aux = {'datos sin NAs en q': qsna, 'Na en q': qna, 'Na en %': ppna}
    return pd.DataFrame(data=aux).sort_values(by='Na en %', ascending=False)


# Valores únicos en columnas categóricas
def val_cat_unicos(df: pd.DataFrame, categorical_cols: list = None) -> None:
    """
    Imprime los valores únicos de columnas categóricas en un DataFrame.
    
    Parámetros:
    ----------
    df : pd.DataFrame
        El DataFrame que contiene los datos.
    categorical_cols : list, opcional
        Lista de nombres de columnas categóricas a analizar. Si no se proporciona, 
        se detectarán automáticamente.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    for col in categorical_cols:
        if col in df.columns:
            print(f"Valores únicos en la columna '{col}':")
            print(df[col].unique())
            print()
        else:
            print(f"La columna '{col}' no existe en el DataFrame.")


# Valores únicos en columnas numéricas
def val_num_unicos(df: pd.DataFrame, numeric_cols: list = None) -> None:
    """
    Imprime los valores únicos de columnas numéricas en un DataFrame.

    Parámetros:
    ----------
    df : pd.DataFrame
        El DataFrame que contiene los datos.
    numeric_cols : list, opcional
        Lista de nombres de columnas numéricas a analizar. Si no se proporciona, 
        se detectarán automáticamente.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if col in df.columns:
            print(f"Valores únicos en la columna '{col}':")
            print(df[col].unique())
            print()
        else:
            print(f"La columna '{col}' no existe en el DataFrame.")


# Detección de valores atípicos (outliers) usando IQR
def detect_outliers_iqr(data: pd.Series) -> pd.Series:
    """
    Detecta valores atípicos en una serie utilizando el rango intercuartílico (IQR).

    Parámetros:
    ----------
    data : pd.Series
        Serie de datos a analizar.

    Retorna:
    -------
    pd.Series
        Una serie booleana indicando True para valores atípicos.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)
