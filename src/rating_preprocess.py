import pandas as pd
from sklearn.model_selection import train_test_split
from consts import RATING_CSV, META_DATA_PARQUET, TRAIN_SET_PARQUET, VALIDATION_SET_PARQUET, TEST_SET_PARQUET

ratings = pd.read_csv(RATING_CSV)
meta_df = pd.read_parquet(META_DATA_PARQUET, engine="fastparquet")

res = (
    ratings.merge(meta_df, on=["movieId"], how="left")
    .dropna()
    .reset_index(drop=True)[["userId", "movieId", "rating"]]
    .astype(str)
)

# train: 80%, validation: 10%, test: 10%
train, valid = train_test_split(res, train_size=0.8)
valid, test = train_test_split(valid, train_size=0.5)

# Save to parquet file
train.to_parquet(TRAIN_SET_PARQUET, engine="fastparquet", compression="gzip")
valid.to_parquet(VALIDATION_SET_PARQUET, engine="fastparquet", compression="gzip")
test.to_parquet(TEST_SET_PARQUET, engine="fastparquet", compression="gzip")