import pandas as pd
from consts import PUNCTUATION, SNOWBALLSTEMMER, STOPWORDS, TRAIN_SET

movies_per_user = (
    pd.read_csv(TRAIN_SET).groupby("userId")["movieId"].apply(set).to_dict()
)


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
    x = x.lower()  # lowercase string
    x = [
        c for c in x if not (c in PUNCTUATION) and (not c.isdigit())
    ]  # non-punct characters
    x = "".join(x)  # convert back to string
    words = x.strip().split()  # tokenizes
    words = [w for w in words if w not in STOPWORDS]  # remove stopwords and numbers
    return " ".join(SNOWBALLSTEMMER.stem(w) for w in words)


def count_recall_at(user, prediction, k):
    top_partition = prediction[:k]
    try:
        watched = movies_per_user.get(user)
        return sum([m in watched for m in top_partition]) / k
    except:
        return 0

