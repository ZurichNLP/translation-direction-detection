from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np
from nmtscore import NMTScorer
from scipy.special import softmax
from scipy.stats import permutation_test


@dataclass
class TranslationDirectionResult:
    sentence1: Union[str, List[str]]
    sentence2: Union[str, List[str]]
    lang1: str
    lang2: str
    raw_prob_1_to_2: float
    raw_prob_2_to_1: float
    pvalue: Optional[float] = None

    @property
    def num_sentences(self):
        return len(self.sentence1) if isinstance(self.sentence1, list) else 1

    @property
    def prob_1_to_2(self):
        return softmax([self.raw_prob_1_to_2, self.raw_prob_2_to_1])[0]

    @property
    def prob_2_to_1(self):
        return softmax([self.raw_prob_1_to_2, self.raw_prob_2_to_1])[1]

    @property
    def predicted_direction(self) -> str:
        if self.raw_prob_1_to_2 >= self.raw_prob_2_to_1:
            return self.lang1 + '→' + self.lang2
        else:
            return self.lang2 + '→' + self.lang1

    def __str__(self):
        s = f"""\
Predicted direction: {self.predicted_direction}
{self.num_sentences} sentence pair{"s" if self.num_sentences > 1 else ""}
{self.lang1}→{self.lang2}: {self.prob_1_to_2:.3f}
{self.lang2}→{self.lang1}: {self.prob_2_to_1:.3f}"""
        if self.pvalue is not None:
            s += f"\np-value: {self.pvalue}\n"
        return s


class TranslationDirectionDetector:

    def __init__(self, scorer: NMTScorer = None, use_normalization: bool = False):
        self.scorer = scorer or NMTScorer()
        self.use_normalization = use_normalization

    def detect(self,
               sentence1: Union[str, List[str]],
               sentence2: Union[str, List[str]],
               lang1: str,
               lang2: str,
               return_pvalue: bool = False,
               pvalue_n_resamples: int = 9999,
               score_kwargs: dict = None
               ) -> TranslationDirectionResult:
        if isinstance(sentence1, list) and isinstance(sentence2, list):
            if len(sentence1) != len(sentence2):
                raise ValueError("Lists sentence1 and sentence2 must have same length")
            if len(sentence1) == 0:
                raise ValueError("Lists sentence1 and sentence2 must not be empty")
            if len(sentence1) == 1 and return_pvalue:
                raise ValueError("return_pvalue=True requires the documents to have multiple sentences")
        if lang1 == lang2:
            raise ValueError("lang1 and lang2 must be different")

        prob_1_to_2 = self.scorer.score_direct(
            sentence2, sentence1,
            lang2, lang1,
            normalize=self.use_normalization,
            both_directions=False,
            score_kwargs=score_kwargs
        )
        prob_2_to_1 = self.scorer.score_direct(
            sentence1, sentence2,
            lang1, lang2,
            normalize=self.use_normalization,
            both_directions=False,
            score_kwargs=score_kwargs
        )
        pvalue = None

        if isinstance(sentence1, list):  # document-level
            # Compute the average probability per target token, across the complete document
            # 1. Convert probabilities back to log probabilities
            log_prob_1_to_2 = np.log2(np.array(prob_1_to_2))
            log_prob_2_to_1 = np.log2(np.array(prob_2_to_1))
            # 2. Reverse the sentence-level length normalization
            sentence1_lengths = np.array([self._get_sentence_length(s) for s in sentence1])
            sentence2_lengths = np.array([self._get_sentence_length(s) for s in sentence2])
            log_prob_1_to_2 = sentence2_lengths * log_prob_1_to_2
            log_prob_2_to_1 = sentence1_lengths * log_prob_2_to_1
            # 4. Sum up the log probabilities across the document
            total_log_prob_1_to_2 = log_prob_1_to_2.sum()
            total_log_prob_2_to_1 = log_prob_2_to_1.sum()
            # 3. Document-level length normalization
            avg_log_prob_1_to_2 = total_log_prob_1_to_2 / sum(sentence2_lengths)
            avg_log_prob_2_to_1 = total_log_prob_2_to_1 / sum(sentence1_lengths)
            # 4. Convert back to probabilities
            prob_1_to_2 = 2 ** avg_log_prob_1_to_2
            prob_2_to_1 = 2 ** avg_log_prob_2_to_1

            if return_pvalue:
                x = np.vstack([log_prob_1_to_2, sentence2_lengths]).T
                y = np.vstack([log_prob_2_to_1, sentence1_lengths]).T
                result = permutation_test(
                    data=(x, y),
                    statistic=self._statistic_token_mean,
                    permutation_type="samples",
                    n_resamples=pvalue_n_resamples,
                )
                pvalue = result.pvalue
        else:
            if return_pvalue:
                raise ValueError("return_pvalue=True requires sentence1 and sentence2 to be lists of sentences")

        return TranslationDirectionResult(
            sentence1=sentence1,
            sentence2=sentence2,
            lang1=lang1,
            lang2=lang2,
            raw_prob_1_to_2=prob_1_to_2,
            raw_prob_2_to_1=prob_2_to_1,
            pvalue=pvalue,
        )

    def _get_sentence_length(self, sentence: str) -> int:
        tokens = self.scorer.model.tokenizer.tokenize(sentence)
        return len(tokens)

    @staticmethod
    def _statistic_token_mean(x: np.ndarray, y: np.ndarray, axis: int = -1) -> float:
        """
        Statistic for scipy.stats.permutation_test

        :param x: Matrix of shape (2 x num_sentences). The first row contains the unnormalized log probability
        for lang1→lang2, the second row contains the sentence lengths in lang2.
        :param y: Same as x, but for lang2→lang1
        :return: Difference between lang1→lang2 and lang2→lang1
        """
        if axis != -1:
            raise NotImplementedError("Only axis=-1 is supported")
        # Add batch dim
        if x.ndim == 2:
            x = x[np.newaxis, ...]
            y = y[np.newaxis, ...]
        # Sum up the log probabilities across the document
        total_log_prob_1_to_2 = x[:, 0].sum(axis=axis)
        total_log_prob_2_to_1 = y[:, 0].sum(axis=axis)
        # Document-level length normalization
        avg_log_prob_1_to_2 = total_log_prob_1_to_2 / x[:, 1].sum(axis=axis)
        avg_log_prob_2_to_1 = total_log_prob_2_to_1 / y[:, 1].sum(axis=axis)
        # Convert to probabilities
        prob_1_to_2 = 2 ** avg_log_prob_1_to_2
        prob_2_to_1 = 2 ** avg_log_prob_2_to_1
        # Compute difference
        return prob_1_to_2 - prob_2_to_1
