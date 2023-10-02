from typing import Optional

import nltk
import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

LIMIT = 20000
SOURCE = "wikipedia"
DATASET = "20220301.simple"


def load_wikipedia_data(source: str, name: str, dataset: Optional[DatasetDict] = None) -> pd.DataFrame:
    if not dataset:
        dataset = load_dataset(source, name)['train']
    
    articles_list = {"text": dataset["text"],
                     "title": dataset["title"]}

    df = pd.DataFrame(articles_list)
    print(df.shape)
    df = df.head(LIMIT)
    return df


if __name__ == '__main__':
    # download from huggingface
    dataset = load_dataset(SOURCE, DATASET)
    print(dataset)
    
    
    # download nltk packages/models for preprocessing
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('punkt')