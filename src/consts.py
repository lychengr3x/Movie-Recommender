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

TRAIN_SET = "../data/train.csv"
VALID_SET = "../data/valid.csv"
