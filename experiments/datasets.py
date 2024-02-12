import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import sacrebleu
from datasets import load_dataset


REF_NAMES = ["ref", "refA", "refB", "refC", "refD", "ref-A", "ref-B", "ref-C", "ref-D", "ref-refA", "ref-refB",
                "ref-refC", "ref-refD"]

WMT2016_NMT_SYSTEMS = ["uedin-nmt", "NYU-UMontreal", "metamind"]


@dataclass
class TranslationExample:
    src: str
    tgt: str
    docid: str
    sysid: str

    @property
    def system_document(self):
        return f"{self.sysid}::{self.docid}"


@dataclass
class TranslationDataset:
    name: str
    type: str  # ht, nmt or pre-nmt
    src_lang: str
    tgt_lang: str
    examples: list[TranslationExample]
    is_indirect: bool = False

    def __str__(self):
        return f"{self.name} ({self.translation_direction}) {self.type}"

    @property
    def num_examples(self) -> int:
        return len(self.examples)

    @property
    def num_sentences(self) -> int:
        sys_id = self.examples[0].sysid
        return len([e for e in self.examples if e.sysid == sys_id])

    @property
    def num_documents(self) -> int:
        sysid = self.examples[0].sysid
        return len(set([e.docid for e in self.examples if e.sysid == sysid]))

    @property
    def num_systems(self) -> int:
        return len(set([e.sysid for e in self.examples]))

    @property
    def num_system_documents(self) -> int:
        return len(set([e.system_document for e in self.examples]))

    @property
    def translation_direction(self):
        return f"{self.src_lang}→{self.tgt_lang}"

    @property
    def lang_pair(self) -> str:
        lang1 = min(self.src_lang, self.tgt_lang)
        lang2 = max(self.src_lang, self.tgt_lang)
        return f"{lang1}↔{lang2}"

    @property
    def documents(self) -> dict[str, list[TranslationExample]]:
        documents = {}
        for example in self.examples:
            if example.docid not in documents:
                documents[example.docid] = []
            documents[example.docid].append(example)
        return documents

    def get_documents(self, min_num_sentences: int = 10) -> dict[str, list[TranslationExample]]:
        return {docid: examples for docid, examples in self.documents.items() if len(examples) >= min_num_sentences}


def load_wmt16_dataset(lang_pair: str, type: str) -> TranslationDataset:
    assert type in ["ht", "nmt", "pre-nmt"]

    # We load src, ref and docid via sacrebleu and sys output via the local data copy
    sacrebleu_dataset = sacrebleu.DATASETS["wmt16"]
    sacrebleu_paths = [Path(p) for p in sacrebleu_dataset.get_files(lang_pair)]
    src_path = [p for p in sacrebleu_paths if p.name.endswith(".src")][0]
    assert src_path.exists()
    ref_paths = list(sorted([p for p in sacrebleu_paths if p.name.split(".")[-1] in REF_NAMES]))
    for ref_path in ref_paths:
        assert ref_path.exists()
    docid_path = [p for p in sacrebleu_paths if p.name.endswith(".docid")][0]
    assert docid_path.exists()
    origlang_path = [p for p in sacrebleu_paths if p.name.endswith(".origlang")][0]
    assert origlang_path.exists()

    src = src_path.read_text().splitlines()
    tgt: List[List[str]] = []
    for ref_path in ref_paths:
        ref = ref_path.read_text().splitlines()
        assert len(ref) == len(src)
        tgt.append(ref)
    docids = docid_path.read_text().splitlines()
    assert len(src) == len(docids)
    origlangs = origlang_path.read_text().splitlines()
    assert len(src) == len(origlangs)

    wmt_dir = Path(__file__).parent / "data" / "wmt16-submitted-data"
    assert wmt_dir.exists()
    src_lang, tgt_lang = lang_pair.split("-")
    systems_dir = wmt_dir / "txt" / "system-outputs" / "newstest2016" / lang_pair
    assert systems_dir.exists()
    systems = list(sorted([f.name for f in systems_dir.iterdir() if f.is_file()]))
    if type == "nmt":
        systems = [s for s in systems if any([infix in s for infix in WMT2016_NMT_SYSTEMS])]
    elif type == "pre-nmt":
        systems = [s for s in systems if not any([infix in s for infix in WMT2016_NMT_SYSTEMS])]
    sys_paths = [systems_dir / s for s in systems]
    sys_outputs: List[List[str]] = []
    for sys_path in sys_paths:
        sys = sys_path.read_text().splitlines()
        assert len(sys) == len(src)
        sys_outputs.append(sys)

    examples = []
    for i in range(len(src)):
        if origlangs[i] != src_lang:
            continue
        if type == "ht":
            for j, ref in enumerate(tgt):
                example = TranslationExample(
                    src=src[i],
                    tgt=ref[i],
                    docid=docids[i],
                    sysid=ref_paths[j].name,
                )
                examples.append(example)
        else:
            for j, sys in enumerate(sys_outputs):
                example = TranslationExample(
                    src=src[i],
                    tgt=sys[i],
                    docid=docids[i],
                    sysid=systems[j],
                )
                examples.append(example)

    return TranslationDataset(
        name=f"wmt16",
        type=type,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        examples=examples,
    )


def load_wmt21_23_dataset(wmt_name: str, lang_pair: str, type: str) -> TranslationDataset:
    assert wmt_name in ["wmt21", "wmt22", "wmt23"]
    assert type in ["ht", "nmt"]

    # We load src, ref and docid via sacrebleu and sys output via the local data copy
    sacrebleu_dataset = sacrebleu.DATASETS[wmt_name]
    sacrebleu_paths = [Path(p) for p in sacrebleu_dataset.get_files(lang_pair)]
    src_path = [p for p in sacrebleu_paths if p.name.endswith(".src")][0]
    assert src_path.exists()
    ref_paths = list(sorted([p for p in sacrebleu_paths if p.name.split(".")[-1] in REF_NAMES]))
    for ref_path in ref_paths:
        assert ref_path.exists()
    docid_path = [p for p in sacrebleu_paths if p.name.endswith(".docid")][0]
    assert docid_path.exists()
    origlang_path = [p for p in sacrebleu_paths if p.name.endswith(".origlang")][0]
    assert origlang_path.exists()

    src = src_path.read_text().splitlines()
    tgt: List[List[str]] = []
    for ref_path in ref_paths:
        ref = ref_path.read_text().splitlines()
        assert len(ref) == len(src)
        tgt.append(ref)
    docids = docid_path.read_text().splitlines()
    assert len(src) == len(docids)
    origlangs = origlang_path.read_text().splitlines()
    assert len(src) == len(origlangs)

    wmt_dir = Path(__file__).parent / "data" / f"{wmt_name}-news-systems"
    assert wmt_dir.exists()
    src_lang, tgt_lang = lang_pair.split("-")
    systems_dir = wmt_dir / "txt" / "system-outputs"
    assert systems_dir.exists()
    if wmt_name == "wmt21":
        system_prefix = "newstest2021"
    elif wmt_name == "wmt22":
        system_prefix = "generaltest2022"
    elif wmt_name == "wmt23":
        system_prefix = "generaltest2023"
    systems = list(sorted([f.name for f in systems_dir.iterdir() if f.is_file() and
               f.name.startswith(f"{system_prefix}.{lang_pair}.hyp.") or
                           f.name.startswith(f"florestest2021.{lang_pair}.hyp.")]))
    for system in systems:
        assert system.split(".")[-1] == tgt_lang
    sys_paths = [systems_dir / s for s in systems]
    sys_outputs: List[List[str]] = []
    for sys_path in sys_paths:
        sys = sys_path.read_text().splitlines()
        assert len(sys) == len(src)
        sys_outputs.append(sys)

    examples = []
    for i in range(len(src)):
        if origlangs[i] != src_lang and src_lang not in {"bn", "hi", "xh", "zu"}:
            continue
        if type == "ht":
            for j, ref in enumerate(tgt):
                example = TranslationExample(
                    src=src[i],
                    tgt=ref[i],
                    docid=docids[i],
                    sysid=ref_paths[j].name,
                )
                examples.append(example)
        else:
            for j, sys in enumerate(sys_outputs):
                example = TranslationExample(
                    src=src[i],
                    tgt=sys[i],
                    docid=docids[i],
                    sysid=systems[j],
                )
                examples.append(example)

    return TranslationDataset(
        name=wmt_name,
        type=type,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        examples=examples,
    )


def load_flores_dataset(lang_pair: str, type: str) -> TranslationDataset:
    assert type in ["ht"]

    # translate to language codes used in flores
    src_lang, tgt_lang = lang_pair.split("-")
    lang_code = {
        "cs": "ces_Latn",
        "uk": "ukr_Cyrl",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "zu": "zul_Latn",
        "xh": "xho_Latn",
        "hi": "hin_Deva",
        "bn": "ben_Beng",
           }

    src_lang_code = lang_code[src_lang]
    tgt_lang_code = lang_code[tgt_lang]
    lang_pair_code = "-".join([src_lang_code, tgt_lang_code])

    # We load src, ref via datasets (although in this case src and ref do not imply translation direction)
    dataset = load_dataset("facebook/flores", lang_pair_code)["devtest"]

    src = [dataset[i]["".join(["sentence_", src_lang_code])] for i in range(len(dataset))]
    tgt = [dataset[i]["".join(["sentence_", tgt_lang_code])] for i in range(len(dataset))]
    docids = [str(dataset[i]["id"]) for i in range(len(dataset))]
    assert len(src) == len(tgt) == len(docids)

    examples = []
    for i in range(len(src)):
        example = TranslationExample(
              src=src[i],
              tgt=tgt[i],
              docid=docids[i],
              sysid='ht', # since there are no systems involved, we chose the 'system name' to be 'ht'
          )
        examples.append(example)

    return TranslationDataset(
        name=f"flores",
        type=type,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        examples=examples,
    )


def load_all_datasets() -> List[TranslationDataset]:
    logging.info("Loading datasets...")
    datasets = []

    # WMT (direct translation between the pair)
    for type in ["ht", "nmt", "pre-nmt"]:
        for lang_pair in ["cs-en", "de-en", "en-cs", "en-de", "en-ru", "ru-en"]:
            datasets.append(load_wmt16_dataset(lang_pair, type))
    for type in ["ht", "nmt"]:
        for lang_pair in ["cs-en", "cs-uk", "de-en", "de-fr", "en-cs", "en-de", "en-ru", "en-uk", "en-zh", "fr-de",
                          "ru-en", "uk-cs", "uk-en", "zh-en"]:
            datasets.append(load_wmt21_23_dataset("wmt22", lang_pair, type))
    for type in ["ht", "nmt"]:
        for lang_pair in ["cs-uk", "en-cs", "en-ru", "en-uk", "en-zh", "ru-en", "uk-en", "zh-en"]:
            datasets.append(load_wmt21_23_dataset("wmt23", lang_pair, type))

    # Flores (both sides are translations from English – translations are not direct between the pair)
    indirect_datasets = []
    lo_res_pairs = ["bn-hi", "hi-bn", "xh-zu", "zu-xh", "cs-uk", "de-fr"]
    for lang_pair in lo_res_pairs:
        indirect_datasets.append(load_flores_dataset(lang_pair, "ht"))
    for dataset in indirect_datasets:
        dataset.is_indirect = True
    datasets += indirect_datasets
    
    return datasets
