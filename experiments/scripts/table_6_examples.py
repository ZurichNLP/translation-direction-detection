from nmtscore import NMTScorer

from translation_direction_detection.detector import TranslationDirectionDetector

model_name = "m2m100_418M"
scorer = NMTScorer(model_name)
detector = TranslationDirectionDetector(scorer)

for src, tgt, src_lang, tgt_lang in [
    ('Mit dem Programm "Guten Tag, liebes Glück" ist er seit 2020 auf Tour.', 'He has been on tour with the programme "Guten Tag, liebes Glück" since 2020.', "de", "en"),
    ('Mit dem Programm "Guten Tag, liebes Glück" ist er seit 2020 auf Tour.', ' He has been on tour since 2020.', "de", "en"),
    ("please try to perfprm thsi procedures\"", "bitte versuchen Sie es mit diesen Verfahren", "en", "de"),
    ("please try to perfprm thsi procedures\"", "Bitte versuchen Sie, diese Prozeduren durchzuführen\"", "en", "de"),
    ("If costs for your country are not listed, please contact us for a quote.", "Wenn die Kosten für Ihr Land nicht aufgeführt sind, wenden Sie sich für einen Kostenvoranschlag an uns.", "en", "de"),
    ("If costs for your country are not listed, please contact us for a quote.", "Wenn die Kosten für Ihr Land nicht aufgeführt sind, kontaktieren Sie uns bitte für ein Angebot.", "en", "de"),
    ("Needless to say, it was chaos.", "Es war natürlich ein Chaos.", "en", "de"),
    ("Needless to say, it was chaos.", "Unnötig zu sagen, es war Chaos.", "en", "de"),
    ("Mit freundlichen Grüßen", "Cordialement", "de", "fr"),
    ("Mit freundlichen Grüßen", "Sincèrement", "de", "fr"),
    ("Mit freundlichen Grüßen", "Sincères amitiés", "de", "fr"),
    ("Mit freundlichen Grüßen", "Avec mes meilleures salutations", "de", "fr"),
]:

    result = detector.detect(src, tgt, src_lang, tgt_lang)
    label = 1 if result.predicted_direction == f"{src_lang}→{tgt_lang}" else 0
    rel_difference = result.raw_prob_1_to_2 / result.raw_prob_2_to_1
    print(src)
    print(tgt)
    print(f"Label {label}")
    print(f"{src_lang}→{tgt_lang}: {result.raw_prob_1_to_2:.3f}")
    print(f"{tgt_lang}→{src_lang}: {result.raw_prob_2_to_1:.3f}")
    print(f"Relative probability difference: {rel_difference:.2f}")
    print()
