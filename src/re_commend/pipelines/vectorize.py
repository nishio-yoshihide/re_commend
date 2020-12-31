from kedro.pipeline import Pipeline, node
from typing import Callable, Dict
import tqdm

import pandas as pd
from transformers import (
    BertModel,
    BertJapaneseTokenizer
)

PRETRAINED_MODEL = 'bert-base-japanese-whole-word-masking'
bert = BertModel.from_pretrained(PRETRAINED_MODEL)
tokenizer = BertJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL)


def create_pipeline(**args):
    return Pipeline([
        node(
            func=vectorize,
            inputs=["brand_details"],
            outputs="brand_vectors"
        ),
        node(
            func=merge,
            inputs=["brands", "brand_details", "brand_vectors"],
            outputs="brand_dataset"
        )
    ])


def vectorize(brand_details: Dict[str, Callable]) -> pd.DataFrame:
    MAX_LENGTH = bert.config.max_position_embeddings - 2

    def _vectorize_each():
        for index, load in brand_details.items():
            detail = load()
            text = detail["review"] + "\n" + detail["description"]

            corpus = tokenizer.encode([b for b in preprocess(text)], max_length = MAX_LENGTH, pad_to_max_length=True, return_tensors='pt')
            yield index, bert_pooling(corpus)

    return pd.DataFrame([
        {"index": int(index), "vector": vector}
        for index, vector in _vectorize_each()
    ])


def preprocess(text):
    """
    文章全体を、文に分ける。urlや記号などは取り除く
    """
    import functools
    # https://qiita.com/wwwcojp/items/3535985007aa4269009c
    from ja_sentence_segmenter.common.pipeline import make_pipeline
    # from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
    from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
    from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

    split_punc2 = functools.partial(split_punctuation, punctuations="。！？")

    segmenter = make_pipeline(
        normalize,
        split_newline,
        split_punc2
    )
    return list(segmenter(text))


def bert_pooling(input_tensor):
    """
    文章全体のベクトルを得る。単語とセンテンスの次元をmeanでつぶす。
    """
    return bert(input_tensor)[0].detach().numpy().mean(axis=(0, 1))


def merge(brands: pd.DataFrame,
          brand_details: Dict[str, Callable],
          brand_vectors: pd.DataFrame) -> pd.DataFrame:
    characters = extract_characters(brand_details)
    return brands.merge(characters, on="index").merge(brand_vectors, on="index")


def extract_characters(brand_details: Dict[str, Callable]) -> pd.DataFrame:
    return pd.DataFrame([
        {"index": int(index), "characters": load()["characters"]}
        for index, load in brand_details.items()
    ])