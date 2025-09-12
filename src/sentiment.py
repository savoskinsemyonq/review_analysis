import argparse, os, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le

def zero_shot_predict(texts, candidate_labels=("positive","neutral","negative")):
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    out = clf(texts, candidate_labels=list(candidate_labels), multi_label=False)
    return [o["labels"][0] for o in out]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/clean_reviews.parquet")
    ap.add_argument("--text-col", default="clean_text")
    ap.add_argument("--label-col", default=None, help="Если нет меток, укажите 'rating' для weak-labels")
    ap.add_argument("--zero-shot", action="store_true", help="Если нет меток — быстрый zero-shot бэйзлайн")
    ap.add_argument("--finetune", action="store_true", help="Включить fine-tuning DistilBERT (дольше)")
    ap.add_argument("--outdir", default="data/artifacts")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)

    if args.label_col is None and "rating" in df.columns:
        args.label_col = "rating"

    if args.zero_shot and (args.label_col is None):
        preds = zero_shot_predict(df[args.text_col].astype(str).tolist())
        df_out = df.copy()
        df_out["sentiment"] = preds
        df_out.to_parquet(f"{args.outdir}/sentiment_zero_shot.parquet", index=False)
        print("Saved zero-shot predictions.")
        return

    if "sentiment" in df.columns:
        y = df["sentiment"]
    else:
        raise SystemExit("Нет меток для обучения. Используйте --zero-shot или укажите колонку с метками.")

    mask = y.notna()
    X_text = df.loc[mask, args.text_col].astype(str).tolist()
    y = y.loc[mask].astype(str).tolist()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X = encoder.encode(X_text, show_progress_bar=True)
    y_enc, le = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    import pickle, json
    with open(f"{args.outdir}/sentiment_lr.pkl","wb") as f: pickle.dump({"clf": clf, "label_encoder": le}, f)
    with open(f"{args.outdir}/sentiment_report.json","w") as f: json.dump(report, f, indent=2)
    print("Saved sentiment model and report.")

if __name__ == "__main__":
    main()
