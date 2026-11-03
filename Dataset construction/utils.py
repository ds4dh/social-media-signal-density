import pandas as pd


def flatten_dict_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Helper function used to preprocess the rough tweet datas, that is expected to be already loaded in a dataframe.
    Specifically, this function flatten dictionary columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the raw tweets data with potential 
        nested structures.

    Returns
    -------
    pd.DataFrame
        A flattened DataFrame where nested fields have been expanded into 
        separate columns.
    """

    filled = df[column].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    expanded = pd.json_normalize(filled)
    expanded.columns = [f"{column}_{c}" for c in expanded.columns]
    expanded.index = df.index
    
    # Replace fully empty rows (originally NaN) with NaN
    expanded[expanded.apply(lambda row: row.astype(bool).sum() == 0, axis=1)] = None
    
    return pd.concat([df.drop(columns=[column]), expanded], axis=1)