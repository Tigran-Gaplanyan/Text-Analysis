import attr
from typing import TypeAlias, Literal

import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError
from nltk import WordNetLemmatizer, PorterStemmer, pos_tag, word_tokenize

pos_refs: TypeAlias = Literal['a', 'r', 'n', 'v']


class Processor:
    def __init__(self, n_stop_words: int = 25, hard_lemmatize: bool = True):
        self.freq = n_stop_words
        self.stemmer = PorterStemmer()
        self.lem = WordNetLemmatizer()
        self.hard_lemmatize = hard_lemmatize

    def process(self, sentece: str, steps: bool = False):
        lemmatized = self.lemmatize_sentence(sentece)
        stemmed = self.stem_sentence(lemmatized)
        lowerised = self.lowerise(stemmed)
        if steps:
            tokenized = self.tokenize(sentece)
            return tokenized, lemmatized, stemmed, lowerised
        return lowerised

    def remove_stop_words(self, tokens: list[str]) -> list[str]:
        top_words = pd.Series(tokens).value_counts()[:self.freq].index.tolist()
        return list(filter(lambda x: x not in top_words, tokens))

    def stem_sentence(self, tokens: list[str]) -> list[str]:
        return list(map(self.stemmer.stem, tokens))

    def lemmatize_sentence(self, sentence: str) -> list[str]:
        """
        Lemmatize words: likes/likely -> like; trader/trading -> trade

        :param word: word subject to be lemmatized
        :return: lemmatized version of the word
        """

        final = []
        pos_tags = self.tag_position(sentence)
        for word, pos_label in pos_tags:
            synset = None
            if pos_label == 'r' and self.hard_lemmatize:  # For adverbs, it's tricky
                try:
                    synset = wordnet.synset(word + '.r.1').lemmas()[0].name()
                except (AttributeError, WordNetError) as e:
                    try:
                        synset = wordnet.synset(word + '.a.1').lemmas()[0].name()
                    except (AttributeError, WordNetError) as e:
                        synset = word
                finally:
                    if synset is None:
                        synset = word

            elif pos_label in ['a', 's', 'v'] and self.hard_lemmatize:  # For adjectives and verbs
                synset = self.lem.lemmatize(word, pos=pos_label)
            else:  # Nltk lemmatizer have nouns as default option
                synset = self.lem.lemmatize(word)
            final.append(synset)
        return final

    @staticmethod
    def tag_position(sentence: str) -> list[tuple[str, pos_refs]]:
        """
        Get the single character pos constant from pos_tag like this:

        pos_refs = {'n': ['NN', 'NNS', 'NNP', 'NNPS'],
                   'v': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                   'r': ['RB', 'RBR', 'RBS'],
                   'a': ['JJ', 'JJR', 'JJS']}

        :param sentence: words subject to be tagged
        :return: shortened tag (key in pos_refs) for each word
        """
        pos_tags = pos_tag(word_tokenize(sentence))
        pos_tags = [(x[0], x[1][0].lower().replace("j", "a")) for x in pos_tags]  # 'j' <--> 'a' reassignment
        return pos_tags

    @staticmethod
    def tokenize(sentence: str) -> list[str]:
        return word_tokenize(sentence)

    @staticmethod
    def lowerise(tokens: list[str]) -> list[str]:
        return list(map(str.lower, tokens))


@attr.s(slots=True)
class DatasetProcessor:
    df: pd.DataFrame = attr.ib(default=None)
    step_by_step: bool = attr.ib(default=True, validator=attr.validators.instance_of(bool))
    processor: Processor = attr.ib(default=Processor)
    processor_kwargs: dict = attr.ib(default={'hard_lemmatize': True, 'n_stop_words': 25},
                                     validator=attr.validators.instance_of(dict))

    def __attrs_post_init__(self):
        self.processor = self.processor(**self.processor_kwargs)
        print('Finished Building DatasetProcessor')

    @df.validator
    def valid_columns(self, attribute, value):
        assert set(value.columns) == {'title', 'text'}, f'{attribute.name} should contain only 2 columns: ' \
                                                             f'text, title, got: {value.columns()}'

    @processor_kwargs.validator
    def valid_kwargs(self, attribute, value):
        remainder = set(value.keys()) - {'hard_lemmatize', 'n_stop_words'}
        if not remainder:
            return
        raise ValueError(f'Error in {attribute.name}: Got extra process_kwargs: {remainder}')

    def __call__(self) -> pd.DataFrame:
        self.df['lowerised'] = self.df.text.apply(lambda x: self.processor.process(x, steps=self.step_by_step))
        if self.step_by_step:
            self.df[['tokenized', 'lemmatized', 'stemmed', 'lowerised']] = \
                pd.DataFrame(self.df.lowerised.tolist(), index=self.df.index)
