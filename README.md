
# Unsupervised Translation Direction Detection

This repository contains the code for the paper "Machine Translation Models are Zero-Shot Detectors of Translation Direction".

**Paper: [http://arxiv.org/abs/2401.06769](http://arxiv.org/abs/2401.06769)**  
**Demo: [https://huggingface.co/spaces/ZurichNLP/translation-direction-detection](https://huggingface.co/spaces/ZurichNLP/translation-direction-detection)**

## Installation
- Requirements: Python >= 3.11, PyTorch
- Clone this repository
- `pip install .`

## Usage

### Analyzing individual sentences

```python
import translation_direction_detection as tdd

detector = tdd.TranslationDirectionDetector()

sentence1 = "Können Sie mir dabei weiter helfen?"
sentence2 = "Pouvez-vous m'aider ?"
lang1 = "de"
lang2 = "fr"

result = detector.detect(sentence1, sentence2, lang1, lang2)
print(result)
# Predicted direction: de→fr
# 1 sentence pair
# de→fr: 0.554
# fr→de: 0.446
```

Note: TranslationDirectionDetector loads the model https://huggingface.co/alirezamsh/small100 by default. Pass an [`nmtscore.NMTScorer` object](https://github.com/ZurichNLP/nmtscore/) to the constructor to use a different model.

### Analyzing documents (= lists of aligned sentences)

```python
import translation_direction_detection as tdd

detector = tdd.TranslationDirectionDetector()

sentences1 = ["Können Sie mir dabei weiter helfen?", "Ja; sehr gerne!"]
sentences2 = ["Pouvez-vous m'aider ?", "Oui, avec plaisir !"]
lang1 = "de"
lang2 = "fr"

result = detector.detect(sentences1, sentences2, lang1, lang2)
print(result)
# Predicted direction: de→fr
# 2 sentence pairs
# de→fr: 0.564
# fr→de: 0.436

# With hypothesis testing (permutation test)
result = detector.detect(sentences1, sentences2, lang1, lang2, return_pvalue=True)
print(result)
# Predicted direction: de→fr
# ...
# p-value: 0.5
```

## Reproducing the experiments from the paper
See [experiments/README.md](experiments/README.md).

## License
MIT License

## Citation
```bibtex
@inproceedings{wastl-etal-2025-machine,
    title = "Machine Translation Models are Zero-Shot Detectors of Translation Direction",
    author = "Wastl, Michelle  and
      Vamvas, Jannis  and
      Sennrich, Rico",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.59/",
    doi = "10.18653/v1/2025.findings-acl.59",
    pages = "1054--1074",
    ISBN = "979-8-89176-256-5",
    abstract = "Detecting the translation direction of parallel text has applications for machine translation training and evaluation, but also has forensic applications, such as resolving plagiarism or forgery allegations. In this work, we explore an unsupervised approach to translation direction detection based on the simple hypothesis that $p(translation|original)>p(original|translation)$, motivated by the well-known simplification effect in translationese or machine-translationese. In experiments with multilingual machine translation models across 20 translation directions, we confirm the effectiveness of the approach for high-resource language pairs, achieving document-level accuracies of 82{--}96{\%} for NMT-produced translations, and 60{--}81{\%} for human translations, depending on the model used."
}
```
