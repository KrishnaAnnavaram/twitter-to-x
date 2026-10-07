import pytest

from twitter_to_x.analysis import prepare, score
from twitter_to_x.config import ACQUISITION_DATE
from twitter_to_x.data.sources import load_csv
from twitter_to_x.data.synthetic import make_posts
from twitter_to_x.sentiment import build_scorer


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "posts.csv"
    make_posts(1500, seed=11).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def prepared(synthetic_csv):
    posts = load_csv(synthetic_csv, salt="test-salt", source="synthetic")
    return prepare(posts, ACQUISITION_DATE, 1500, seed=11)


@pytest.fixture(scope="session")
def scored(prepared):
    df, _ = prepared
    return score(df, build_scorer("lexicon"))
