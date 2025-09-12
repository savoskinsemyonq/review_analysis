import argparse, re, os, pandas as pd, numpy as np
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

try:
    import spacy
except ImportError:
    spacy = None

nltk.download('stopwords', quiet=True)

CLEAN_COL = "clean_text"

def basic_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-zА-Яа-я0-9\s\-\']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_lemmatizer(lang):
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm" if lang=="en" else "ru_core_news_sm", disable=["ner","parser"])
            return ("spacy", nlp)
        except Exception:
            pass
    stemmer = SnowballStemmer("english" if lang=="en" else "russian")
    sw = set(stopwords.words('english' if lang=='en' else 'russian'))
    return ("nltk", (stemmer, sw))

def process_row(txt, lemmatizer, lang):
    mode, obj = lemmatizer
    if mode == "spacy":
        doc = obj(basic_clean(txt))
        toks = [t.lemma_.lower() for t in doc if not (t.is_stop or t.is_punct or t.like_num or len(t) < 2)]
    else:
        stemmer, sw = obj
        toks = [stemmer.stem(t.lower()) for t in basic_clean(txt).split() if t.lower() not in sw and len(t)>1]
    return " ".join(toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--lang", default="en", choices=["en","ru"])
    ap.add_argument("--output", default="data/clean_reviews.parquet")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    assert "text" in df.columns, "CSV должен содержать столбец 'text'"
    lemm = build_lemmatizer(args.lang)
    df[CLEAN_COL] = df["text"].astype(str).apply(lambda x: process_row(x, lemm, args.lang))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved -> {args.output}  ({len(df)} rows)")

if __name__ == "__main__":
    main()
