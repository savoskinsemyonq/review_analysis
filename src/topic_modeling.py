import argparse, os, json, pandas as pd, numpy as np
from pathlib import Path

def run_lda(df, text_col, num_topics, min_df, max_df, passes, random_state):
    from gensim import corpora, models
    texts = [row.split() for row in df[text_col].astype(str).tolist()]
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=min_df, no_above=max_df)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=random_state)
    topics = lda.print_topics(num_topics=num_topics, num_words=10)
    doc_topics = [max(lda[corpus[i]], key=lambda x: x[1])[0] if len(lda[corpus[i]])>0 else -1 for i in range(len(corpus))]
    return lda, dictionary, corpus, topics, doc_topics

def run_bertopic(df, text_col, low_memory):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    docs = df[text_col].astype(str).tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True)
    topic_model = BERTopic(low_memory=low_memory, calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/clean_reviews.parquet")
    ap.add_argument("--text-col", default="clean_text")
    ap.add_argument("--method", choices=["lda","bertopic"], required=True)
    ap.add_argument("--num-topics", type=int, default=10)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.5)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--low-memory", action="store_true")
    ap.add_argument("--outdir", default="data/artifacts")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)

    if args.method == "lda":
        lda, dictionary, corpus, topics, doc_topics = run_lda(df, args.text_col, args.num_topics, args.min_df, args.max_df, args.passes, args.random_state)
        import pickle, pyLDAvis.gensim_models as gensimvis, pyLDAvis
        with open(f"{args.outdir}/lda_model.pkl","wb") as f: pickle.dump(lda, f)
        dictionary.save(f"{args.outdir}/lda_dictionary.dict")
        with open(f"{args.outdir}/lda_topics.json","w") as f: json.dump({i:str(t) for i,t in enumerate(topics)}, f, ensure_ascii=False, indent=2)
        df_out = df.copy()
        df_out["topic_lda"] = doc_topics
        df_out.to_parquet(f"{args.outdir}/lda_doc_topics.parquet", index=False)
        vis = gensimvis.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(vis, f"{args.outdir}/lda_vis.html")
        print("LDA artifacts saved.")
    else:
        topic_model, topics, probs = run_bertopic(df, args.text_col, args.low_memory)
        import pickle
        with open(f"{args.outdir}/bertopic_model.pkl","wb") as f: pickle.dump(topic_model, f)
        df_out = df.copy()
        df_out["topic_bertopic"] = topics
        df_out.to_parquet(f"{args.outdir}/bertopic_doc_topics.parquet", index=False)
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(f"{args.outdir}/bertopic_topics.html")
        except Exception as e:
            print("Visualization failed:", e)
        print("BERTopic artifacts saved.")

if __name__ == "__main__":
    main()
