from typing import List

from experiments.datasets import TranslationDataset
from translation_direction_detection.detector import TranslationDirectionDetector

nllb_language_codes = {
    "bn": "ben_Beng",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "ha": "hau_Latn",
    "hi": "hin_Deva",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "xh": "xho_Latn",
    "zh": "zho_Hans",  # simplified
    "zu": "zul_Latn"
}


def lang(lang_code, model_name: str) -> str:
    if "nllb" in model_name.lower():
        return nllb_language_codes[lang_code]
    else:
        return lang_code


def evaluate_sentence_level(detector: TranslationDirectionDetector, datasets: List[TranslationDataset]) -> float:
    num_total = 0
    num_correct = 0
    for dataset in datasets:
        src_lang = lang(dataset.src_lang, str(detector.scorer.model))
        tgt_lang = lang(dataset.tgt_lang, str(detector.scorer.model))
        for example in dataset.examples:
            result = detector.detect(
                sentence1=example.src,
                sentence2=example.tgt,
                lang1=src_lang,
                lang2=tgt_lang,
            )
            num_total += 1
            if result.predicted_direction == f"{src_lang}→{tgt_lang}":
                num_correct += 1
    accuracy = num_correct / num_total * 100
    return accuracy


def evaluate_document_level(
        detector: TranslationDirectionDetector,
        datasets: List[TranslationDataset],
        min_num_sentences=10,
) -> float:
    if min_num_sentences < 2:
        raise ValueError("min_num_sentences must be at least 2")
    num_total = 0
    num_correct = 0
    for dataset in datasets:
        src_lang = lang(dataset.src_lang, str(detector.scorer.model))
        tgt_lang = lang(dataset.tgt_lang, str(detector.scorer.model))
        for document in dataset.get_documents(min_num_sentences=min_num_sentences).values():
            result = detector.detect(
                sentence1=[example.src for example in document],
                sentence2=[example.tgt for example in document],
                lang1=src_lang,
                lang2=tgt_lang,
            )
            num_total += 1
            is_correct = result.predicted_direction == f"{src_lang}→{tgt_lang}"
            num_correct += is_correct
    accuracy = num_correct / num_total * 100
    return accuracy
