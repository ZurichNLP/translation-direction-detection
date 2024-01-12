import json
from functools import cached_property
from pathlib import Path
from typing import Dict, Union, List

from sqlitedict import SqliteDict


class CachedTranslationModel:
    """Loads the complete nmtscore score cache into memory at once."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.score_cache: Dict[str, float] = {}
        cache_parent_dir = Path(__file__).parent / "cached_scores"
        assert cache_parent_dir.exists()
        cache_dirs = list(cache_parent_dir.glob(f"nmtscore_cache*"))
        db_paths = []
        for cache_dir in cache_dirs:
            db_path = cache_dir / f"{model_name.replace('/', '_')}.sqlite"
            if db_path.exists():
                db_paths.append(db_path)
        if not db_paths:
            raise FileNotFoundError(f"Could not find any cache for model {model_name} in {cache_parent_dir}")
        for db_path in sorted(db_paths):
            try:
                with SqliteDict(db_path, timeout=15, encode=json.dumps, decode=json.loads) as db:
                    items = list(db.items())
            except Exception as e:
                print(f"Could not load {db_path} due to {e}")
                continue
            else:
                self.score_cache.update(items)

    def __str__(self):
        return f"CachedTranslationModel({self.model_name})"

    def score(self,
              tgt_lang: str,
              source_sentences: Union[str, List[str]],
              hypothesis_sentences: Union[str, List[str]],
              src_lang: str = None,
              batch_size: int = None,
              use_cache: bool = True,
              **kwargs,
              ) -> Union[float, List[float]]:
        if not use_cache:
            raise NotImplementedError("CachedTranslationModel does not support scoring without cache.")
        assert type(source_sentences) == type(hypothesis_sentences)
        if isinstance(source_sentences, str):
            source_sentences_list = [source_sentences]
            hypothesis_sentences_list = [hypothesis_sentences]
        elif isinstance(source_sentences, list):
            assert len(source_sentences) == len(hypothesis_sentences)
            source_sentences_list = source_sentences
            hypothesis_sentences_list = hypothesis_sentences
        else:
            raise ValueError

        scores_list = []
        for source_sentence, hypothesis_sentence in zip(source_sentences_list, hypothesis_sentences_list):
            key = f"{(src_lang + '_') if src_lang is not None else ''}{tgt_lang}_score_{source_sentence}_{hypothesis_sentence}"
            score = self.score_cache.get(key, None)
            if score is None:
                raise KeyError(f"Could not find cached score for key {key}")
            scores_list.append(score)

        if isinstance(source_sentences, str):
            scores = scores_list[0]
        else:
            scores = scores_list
        return scores

    def translate(self, *args, **kwargs):
        raise NotImplementedError("CachedTranslationModel does not support translation.")

    @property
    def requires_src_lang(self) -> bool:
        if "small100" in self.model_name:
            return False
        elif "m2m100" in self.model_name:
            return True
        elif "nllb" in self.model_name:
            return True
        else:
            raise NotImplementedError

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_name)
