"""The two lexical baselines: 22-cue keyword matcher and TF-IDF + logistic regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from cues import has_cue, mask_cues, normalize


class CueMatcher:
    """Predict NK (1) if any of the 22 search cues occurs in the text, else K/N (0).

    No training. Masking the cues would make it predict K/N for everything,
    so there is no cue-masked variant of this baseline.
    """

    name = "22-cue matcher"

    def predict(self, texts):
        return [int(has_cue(t)) for t in texts]


class TfidfLR:
    """Word 1-3-gram TF-IDF + logistic regression, trained on the same v8 synthetic data.

    Tests whether contextual encoders beat a model that only sees local n-grams.
    """

    name = "TF-IDF + LR"

    def __init__(self, seed: int = 42):
        self.model = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 3),
                min_df=2,
                sublinear_tf=True,
                lowercase=True,
                token_pattern=r"(?u)\b\w[\w']*\b",  # keep "don't" as one token
            ),
            LogisticRegression(max_iter=2000, random_state=seed),
        )

    def fit(self, texts, labels):
        self.model.fit([normalize(t) for t in texts], labels)
        return self

    def predict(self, texts, mask: bool = False):
        texts = [normalize(t) for t in texts]
        if mask:  # no mask token in TF-IDF, so cue words are deleted instead
            texts = [mask_cues(t, "") for t in texts]
        return self.model.predict(texts).tolist()
