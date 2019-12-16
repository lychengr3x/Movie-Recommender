from nltk.stem.snowball import SnowballStemmer
import string
import nltk
from nltk.corpus import stopwords

PUNCTUATION = string.punctuation
SNOWBALLSTEMMER = SnowballStemmer("english")
try:
    STOPWORDS = stopwords.words("english")
except:
    nltk.download("stopwords")
    STOPWORDS = stopwords.words("english")

RATING_CSV = "../data/ratings.csv"
META_DATA_PARQUET = "../data/processed_data.parquet"
TRAIN_SET_PARQUET = "../data/rating_train.parquet"
VALIDATION_SET_PARQUET = "../data/rating_valid.parquet"
TEST_SET_PARQUET = "../data/rating_test.parquet"
