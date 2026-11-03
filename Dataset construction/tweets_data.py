
import json
import glob
import pandas as pd
from constants import *
from utils import *
import numpy as np


def construct_tweets_data():
    """
    Main function used to construct the tweets data. This function will handle the entire process of
    fetching, cleaning, and structuring the tweets data for further analysis or modeling.
    """

    # load the tweets data from the source files
    df = load_tweets_data()

    # flattent dictionary columns
    df = flatten_tweets_data(df)

    # avoid the tweet located in france located with localisation longer than 350km from Geneva
    df = filter_tweets_by_location(df, max_distance_km=350)

    # compute the number dataset in the final df
    print(f"Final dataset contains {len(df)} tweets.")

    # save the df as pkl for exploratory data analyis
    df.to_pickle(DATA_DIR / "tweets.pkl")

    return df


def filter_tweets_by_location(df: pd.DataFrame, max_distance_km: float = 350) -> pd.DataFrame:
    """
    Filter tweets based on their geographical location relative to Geneva. 
    Specifically, this function will keep tweets that either do not have coordinates (i.e. they are in Switzerland) or are located at least `max_distance_km` kilometers away from Geneva.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the tweets data, which should include a 'geo_coordinates.coordinates' column with [longitude, latitude] format.
    max_distance_km : float, optional
        The maximum distance in kilometers from Geneva for a tweet to be retained if it has coordinates.
    """

    # Identify rows with valid coordinates
    has_coords = df['geo_coordinates.coordinates'].notna()

    # Calculate distances only for rows that have them
    distances = df[has_coords].apply(calculate_dist, axis=1)
    
    # Keep if (No Coordinates) OR (Distance <= max_distance_km)
    mask = ~has_coords | (distances <= max_distance_km)
    
    # Reindex mask to match original df length (filling missing with True)
    mask = mask.reindex(df.index, fill_value=True)
    
    return df[mask]



def calculate_dist(row):
    """
    Utility function to calculate the distance from a tweet's location to Geneva using the Haversine formula.
    """


    # Twitter API format is [longitude, latitude]
    lon2, lat2 = row['geo_coordinates.coordinates']
        
    # Haversine Formula
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - LAT_G) * p)/2 + \
    np.cos(LAT_G * p) * np.cos(lat2 * p) * \
    (1 - np.cos((lon2 - LON_G) * p)) / 2
        
    return 12742 * np.asin(np.sqrt(a)) # 2 * R; R = 6371 km

def flatten_tweets_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the nested structure of the tweets data.

    This function takes a DataFrame containing tweets data, which may have 
    nested structures (e.g., geo, entities, public_metrics), and flattens 
    it into a more tabular format suitable for analysis.

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
    df = df.copy()


    # ------------------------------------------------
    # Flatten simple dictionary columns
    # ------------------------------------------------
    dict_columns = ["public_metrics", "edit_controls", "attachments"]
    for col in dict_columns:
        if col in df.columns:
            df = flatten_dict_column(df, col)

    # ------------------------------------------------
    # Flatten geo separately due to its nested 
    # coordinates structure
    # ------------------------------------------------
    if "geo" in df.columns:
        df = flatten_dict_column(df, "geo")
        # geo_coordinates is itself a dict {'type': 'Point', 'coordinates': [lon, lat]}
        # flatten it manually into two clean columns
        if "geo_coordinates" in df.columns:
            df["geo_longitude"] = df["geo_coordinates"].apply(
                lambda x: x["coordinates"][0] if isinstance(x, dict) else None
            )
            df["geo_latitude"] = df["geo_coordinates"].apply(
                lambda x: x["coordinates"][1] if isinstance(x, dict) else None
            )
            df = df.drop(columns=["geo_coordinates"])
        # Drop the redundant coordinates type column if present
        if "geo_coordinates_type" in df.columns:
            df = df.drop(columns=["geo_coordinates_type"])

    # ------------------------------------------------
    # Extract information from referenced_tweets
    # Only the first reference is extracted as tweets
    # rarely have more than one
    # ------------------------------------------------
    if "referenced_tweets" in df.columns:
        df["referenced_tweet_id"] = df["referenced_tweets"].apply(
            lambda x: x[0]["id"] if isinstance(x, list) and len(x) > 0 else None
        )
        df["referenced_tweet_type"] = df["referenced_tweets"].apply(
            lambda x: x[0]["type"] if isinstance(x, list) and len(x) > 0 else None
        )
        df = df.drop(columns=["referenced_tweets"])

    # ------------------------------------------------
    # Extract fields from entities
    # ------------------------------------------------
    if "entities" in df.columns:
        df["hashtags"] = df["entities"].apply(
            lambda x: [h["tag"] for h in x["hashtags"]]
            if isinstance(x, dict) and "hashtags" in x else None
        )
        df["urls"] = df["entities"].apply(
            lambda x: [u["expanded_url"] for u in x["urls"]]
            if isinstance(x, dict) and "urls" in x else None
        )
        df["mentions"] = df["entities"].apply(
            lambda x: [m["username"] for m in x["mentions"]]
            if isinstance(x, dict) and "mentions" in x else None
        )
        df = df.drop(columns=["entities"])

    # ------------------------------------------------
    # Extract domain names from context_annotations
    # ------------------------------------------------
    if "context_annotations" in df.columns:
        df["context_domains"] = df["context_annotations"].apply(
            lambda x: list({a["domain"]["name"] for a in x})
            if isinstance(x, list) and len(x) > 0 else None
        )
        df = df.drop(columns=["context_annotations"])

    # ------------------------------------------------
    # Convert list fields to strings
    # ------------------------------------------------
    if "edit_history_tweet_ids" in df.columns:
        df["edit_history_tweet_ids"] = df["edit_history_tweet_ids"].astype(str)

    return df


def load_tweets_data() -> pd.DataFrame:
    """
    Load and concatenate tweet data from multiple files.

    Specifically, this function loads all '*.jsonl' files located in the
    './Rough_data' directory and concatenates them into a single DataFrame
    for further processing.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all tweets from the loaded files.

    Raises
    ------
    FileNotFoundError
        If no '.jsonl' files are found in the './Rough_data' directory.
    """

    jsonl_files = list(DATA_DIR.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError("No '.jsonl' files found.")

    tweets = []

    for file in jsonl_files:

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                data = obj.get("data")

                if isinstance(data, list):
                    tweets.extend(data)

                elif isinstance(data, dict):
                    tweets.append(data)

    df = pd.DataFrame(tweets)

    return df





