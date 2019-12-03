import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from utils import Jaccard, nlp

class ContentBasedFiltering:
    def __init__(self, meta_file):
        assert isinstance(meta_file, str)
        assert meta_file.endswith("parquet")

        self.__meta_df = pd.read_parquet(meta_file, engine="fastparquet").reset_index(
            drop=True
        )
        self.__movieId_to_index = (
            self.__meta_df["movieId"].reset_index().set_index("movieId").squeeze()
        )
        self.__movieId_to_title = (
            self.__meta_df[["title", "movieId"]].set_index("movieId").squeeze()
        )
        self.__index_to_movieId = self.meta_df["movieId"]

        self.__tfidf = TfidfVectorizer(analyzer="word", stop_words="english", ngram_range=(1,3))
        self.__tfidf_matrix = None
        self.__cosine_sim = None
        self.__jaccard_sim = None

    @property
    def meta_df(self):
        return self.__meta_df

    @property
    def movieId_to_index(self):
        return self.__movieId_to_index

    @property
    def movieId_to_title(self):
        return self.__movieId_to_title
    
    @property
    def index_to_movieId(self):
        return self.__index_to_movieId
        
    @property
    def tfidf(self):
        return self.__tfidf

    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def cosine_sim(self):
        return self.__cosine_sim

    @property
    def jaccard_sim(self):
        return self.__jaccard_sim

    def compute_cosine_similarity(self):
        """
        Compute cosine similarity based on movie description, including overview and tagline.
        """
        if self.cosine_sim is not None:
            return None
        else:
            self.__meta_df["description"] = (
                self.__meta_df["overview"] + " " + self.__meta_df["tagline"]
            )
            self.__meta_df.loc[:, "description"] = self.__meta_df.loc[
                :, "description"
            ].apply(nlp)
            # tfidf matrix where row represents each movie and column represents words
            self.__tfidf_matrix = self.__tfidf.fit_transform(self.__meta_df["description"])
            self.__cosine_sim = linear_kernel(self.__tfidf_matrix)

    def compute_jaccard_similarity(self, fname="JACCARD_SIM.npz"):
        """
        Compute jaccard similarity based on movie meta data, including cast, keywords, genres, director

        Args:
            fname (str)
        """
        if self.jaccard_sim is not None:
            return None
        else:
            if not fname:  # time-consuming
                self.meta_df["items_for_jaccard"] = (
                    self.meta_df["cast"]
                    + self.meta_df["keywords"]
                    + self.meta_df["genres"]
                    + self.meta_df["director"]
                )
                jaccard_sim = np.zeros((len(self.meta_df.index), len(self.meta_df.index)))
                for i1 in tqdm(range(len(self.meta_df.index))):
                    for i2 in range(i1 + 1, len(self.meta_df.index)):
                        s1 = set(self.meta_dfa_new.items_for_jaccard[i1])
                        s2 = set(self.meta_df.items_for_jaccard[i2])
                        try:
                            sim = Jaccard(s1, s2)
                        except:
                            print(f"Jaccard has trouble: (s1, s2) = ({s1}, {s2})")
                            sim = 0
                        jaccard_sim[i1, i2] = sim
                        jaccard_sim[i2, i1] = sim
                self.__jaccard_sim = jaccard_sim
            else:
                assert isinstance(fname, str)
                assert fname.endswith(".npz")
                # require large memory
                self.__jaccard_sim = np.load(fname)["arr_0"]  # shape: (45116, 45116)

    def recommend(self, movie_id, topk=30):
        """
        Recommend movie based on `movie_id`
        """
        assert isinstance(movie_id, int)
        assert isinstance(topk, int)
        assert self.cosine_sim is not None
        assert self.jaccard_sim is not None

        index = self.__movieId_to_index[movie_id]
        sim = self.__cosine_sim[index] + self.__jaccard_sim[index]
        sim = np.argsort(-sim)[1:topk+1]
        sim = [str(self.__index_to_movieId[i]) for i in sim]
        return sim

