
# Unsupervised Translation Direction Detection

This repository contains the code for the paper "Machine Translation Models are Zero-Shot Detectors of Translation Direction".

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
tba
```
