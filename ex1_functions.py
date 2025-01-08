import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

################################
# 	Load the Dataset (10 points)
################################
def load_data(filepath: str) -> pd.DataFrame:
    path_split = filepath.split('.')
    if path_split[-1]=='csv':
        return pd.read_csv(filepath,encoding = 'UTF-8')
    return pd.read_excel(filepath)



########################################
# 	Group and Aggregate Data (20 points)
########################################
def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    return df.groupby(group_by_column).agg(agg_func).reset_index()



#######################################
# 	Remove Sparse Columns (10 points)
#######################################
def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
     # Separate string columns and numeric columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    numeric_cols = df.select_dtypes(exclude=['object', 'string'])
    # Filter numeric columns based on threshold
    numeric_cols_filtered = numeric_cols.loc[:, numeric_cols.sum() >= threshold]
    # Combine the numeric and string columns
    df_filtered = pd.concat([df[string_cols],numeric_cols_filtered], axis=1)
    return df_filtered



#################################################
# 	Dimensionality Reduction with PCA (30 points)
#################################################
def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    # Separate data into Y(dependent) and X (features)
    y = df[meta_columns]
    x = df.drop(meta_columns, axis='columns').fillna(0).copy()

    # normalize the data (for each feature)
    for col in x.columns:
        x[col] = (x[col] - x[col].mean()) / x[col].std(ddof=0)  # שימוש ב-ddof=0 גם בנרמול

    # convert to numpy array for easier calcs
    x_np = x.to_numpy()

    # חישוב קווריאנס של האוכלוסייה (לא של מדגם)
    x_cov = np.cov(x_np.T, ddof=0)

    # compute egenvectors and egenvalues
    eigenvalues, eigenvectors = np.linalg.eig(x_cov)

    # zip, and sort by eigenvalues(x[0]) decending
    eigenvalue_vector_pairs_sorted = sorted(list(zip(eigenvalues, eigenvectors.T)), key=lambda x: x[0], reverse=True)

    # unzip
    sorted_eigenvalues = np.array([x[0] for x in eigenvalue_vector_pairs_sorted])
    sorted_eigenvectors = np.array([x[1] for x in eigenvalue_vector_pairs_sorted])

    # fix sorted_eigenvectors to abs
    for i in range(sorted_eigenvectors.shape[0]):
        if np.max(np.abs(sorted_eigenvectors[i])) != np.max(sorted_eigenvectors[i]):
            sorted_eigenvectors[i] *= -1

    # create matrix P (matrix of eigenvectors) in size num_components
    P = sorted_eigenvectors[:num_components, :].T

    # calc x_np*P to get projection of the original data onto the new feature space defined by the eigenvectors
    new_df = np.dot(x_np, P)

    # Convert the projection data into a DataFrame and name columns as PC1, PC2, etc.
    results_df = pd.DataFrame(data=new_df, columns=[f"PC{i + 1}" for i in range(num_components)])

    # restore meta_columns (dependent values - y)
    results_df[meta_columns] = y

    return results_df

