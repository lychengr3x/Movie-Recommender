import pandas as pd
from consts import (
    PUNCTUATION,
    SNOWBALLSTEMMER,
    STOPWORDS,
    META_DATA_PARQUET,
    TRAIN_SET_PARQUET,
    VALIDATION_SET_PARQUET,
    TEST_SET_PARQUET,
)


def read_meta_data():
    return pd.read_parquet(META_DATA_PARQUET, engine="fastparquet")


def read_training_data():
    return pd.read_parquet(TRAIN_SET_PARQUET, engine="fastparquet")


def read_validation_data():
    return pd.read_parquet(VALIDATION_SET_PARQUET, engine="fastparquet")


def read_testing_data():
    return pd.read_parquet(TEST_SET_PARQUET, engine="fastparquet")


def get_movies_per_user(df):
    """
    Get itemset, i.e. movies per user

    Args:
        df (pd.DataFrame)
    """
    assert isinstance(df, type(pd.DataFrame()))
    return df.groupby("userId")["movieId"].apply(set).to_dict()


def Jaccard(s1, s2):
    assert isinstance(s1, set)
    assert isinstance(s2, set)
    nom = len(s1.intersection(s2))
    den = len(s1.union(s2))
    return nom / den


def nlp(x):
    """
        Natural Language Processing
        
        Args:
            x (str)
        
        Returns:
            (str)
        """
    assert isinstance(x, str)
    # lowercase string
    x = x.lower()
    # non-punct characters
    x = [c for c in x if not (c in PUNCTUATION) and (not c.isdigit())]
    # convert back to string
    x = "".join(x)
    # tokenizes
    words = x.strip().split()
    # remove stopwords and numbers
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(SNOWBALLSTEMMER.stem(w) for w in words)


def recall_at_k(user, prediction, k, items_per_user):
    """
    Compute recall at k.

    Note: data type of items in prediction should be the same as those in items_per_user.
    In my pre-processed data, they are string format.
    """
    assert isinstance(user, str)
    assert isinstance(prediction, list)
    assert isinstance(k, int)
    assert isinstance(items_per_user, dict)
    top_partition = prediction[:k]
    try:
        watched = items_per_user.get(user)
        return sum([m in watched for m in top_partition]) / k
    except:
        return 0

