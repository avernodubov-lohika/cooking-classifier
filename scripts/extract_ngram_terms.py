"""Extract cooking-related unigram/bigram terms from the cooking-rose typo dictionaries.

    python3 scripts/extract_ngram_terms.py [UNIGRAMS_CSV] [BIGRAMS_CSV]

Inputs default to cooking-rose/resources/com/nytimes/cooking/typo/{unigrams,bigrams}.csv
(sibling checkout). Writes data/ngram_safe_unigrams.txt and data/ngram_safe_bigrams.txt, which
generate_dataset.py samples as SAFE_COOKING examples.

The dictionaries are built from the whole recipe index (instructions included), so they are
noisy: stop words, step fragments ("meanwhile set"), person names. Heuristics:
  * unigrams: alphabetic, >= 4 letters, not a stop/instruction word, freq >= MIN_FREQ_UNIGRAM,
    and a cooking vocab word or a partner token in a kept bigram
  * bigrams: both tokens clean AND at least one token is a known ingredient/technique/dish/diet
    word from generate_dataset.py, freq >= MIN_FREQ_BIGRAM
Review the output before committing; names and generic words can still slip through.
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_dataset as gd  # noqa: E402

DEFAULT_DIR = (Path(__file__).resolve().parents[2]
               / "cooking-workspace/cooking-rose/resources/com/nytimes/cooking/typo")
MIN_FREQ_UNIGRAM = 50
MIN_FREQ_BIGRAM = 50

STOP = set("""a an the and or of to in on at for with from by as is are was were be been it its this
that these those you your we our they their he she his her them then than so if but not no nor into
onto over under about up down out off again once only own same too very can will just do does did
done have has had having more most some such any each few both other all also until while when where
which who whom how what why per via etc one two three four five six ten new make made like well
good best""".split())
INSTRUCTION = set("""add bake cook stir set heat remove cover mix combine place put let make serve
transfer pour reduce continue season taste minutes minute hour hours degrees preheat meanwhile
remaining reserved prepare prepared ingredients ahead use used using need needed""".split())
WORD = re.compile(r"[a-zà-ÿ][a-zà-ÿ'-]*[a-zà-ÿ]")


def clean(w, min_len=4):
    return bool(WORD.fullmatch(w)) and len(w) >= min_len and w not in STOP and w not in INSTRUCTION


def anchor_words():
    words = set()
    for term in gd.INGREDIENTS + gd.TECHNIQUES + gd.DISHES + gd.DIETS:
        for w in re.findall(r"[a-zà-ÿ]+", term.lower()):
            if len(w) >= 3:
                words.add(w)
                words.add(w + "s")
                words.add(w.rstrip("s"))
    return words - STOP


def read(path):
    with open(path, encoding="utf-8") as f:
        return [(r["term"], int(r["freq"])) for r in csv.DictReader(f)]


def main():
    uni_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR / "unigrams.csv"
    bi_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DIR / "bigrams.csv"
    anchors = anchor_words()

    candidates = [w for w, f in read(uni_path) if f >= MIN_FREQ_UNIGRAM and clean(w)]
    bigrams = []
    for term, f in read(bi_path):
        if f < MIN_FREQ_BIGRAM:
            continue
        a, b = term.split(" ")
        if clean(a, 3) and clean(b, 3) and (a in anchors or b in anchors):
            bigrams.append(term)

    # Keep a unigram only if it is a cooking vocab word or pairs with one in a kept bigram;
    # this drops generic words ("much", "never", "found") that the frequency cut alone lets in.
    partners = {w for term in bigrams for w in term.split(" ")}
    unigrams = [w for w in candidates if w in anchors or w in partners]

    out = gd.DATA_DIR
    (out / "ngram_safe_unigrams.txt").write_text("\n".join(unigrams) + "\n", encoding="utf-8")
    (out / "ngram_safe_bigrams.txt").write_text("\n".join(bigrams) + "\n", encoding="utf-8")
    print(f"unigrams={len(unigrams)} bigrams={len(bigrams)}")


if __name__ == "__main__":
    main()
