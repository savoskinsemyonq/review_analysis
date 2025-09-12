
import streamlit as st
import pandas as pd
import numpy as np
import json, os, pickle
import plotly.express as px

st.set_page_config(page_title="NLP Отзывы: Темы и Сентимент", layout="wide")

st.title("🔎 Анализ отзывов: Темы × Сентимент")

data_file = st.sidebar.text_input("Путь к parquet с предобработкой", "data/clean_reviews.parquet")
art_dir   = st.sidebar.text_input("Каталог артефактов", "data/artifacts")
method    = st.sidebar.selectbox("Метод тем", ["lda","bertopic"])

@st.cache_data(show_spinner=False)
def load_df(path):
    return pd.read_parquet(path)

def load_topics(method, art_dir):
    if method == "lda":
        topics_path = os.path.join(art_dir, "lda_doc_topics.parquet")
        if os.path.exists(topics_path):
            df_topics = pd.read_parquet(topics_path)
            return df_topics, "topic_lda"
    else:
        topics_path = os.path.join(art_dir, "bertopic_doc_topics.parquet")
        if os.path.exists(topics_path):
            df_topics = pd.read_parquet(topics_path)
            return df_topics, "topic_bertopic"
    return None, None

df = load_df(data_file)
df_topics, topic_col = load_topics(method, art_dir)
if df_topics is not None:
    df = df.merge(df_topics[[topic_col]], left_index=True, right_index=True, how="left")

# Sentiment load (optional)
sent_path = os.path.join(art_dir, "sentiment_report.json")
sent_model_path = os.path.join(art_dir, "sentiment_lr.pkl")
sent_available = os.path.exists(sent_path) and os.path.exists(sent_model_path)

st.sidebar.markdown("---")
if "date" in df.columns:
    min_d, max_d = df["date"].min(), df["date"].max()
    d_range = st.sidebar.date_input("Диапазон дат", [min_d, max_d])
    if isinstance(d_range, list) and len(d_range) == 2:
        df = df[(df["date"] >= pd.to_datetime(d_range[0])) & (df["date"] <= pd.to_datetime(d_range[1]))]


if "split" in df.columns:
    splits = ["(all)"] + sorted(df["split"].dropna().astype(str).unique().tolist())
    sel_split = st.sidebar.selectbox("Фильтр по split (train/test)", splits)
    if sel_split != "(all)":
        df = df[df["split"].astype(str) == sel_split]


if "product_id" in df.columns:
    prods = ["(все)"] + sorted(df["product_id"].dropna().astype(str).unique().tolist())
    sel = st.sidebar.selectbox("Фильтр по продукту", prods)
    if sel != "(все)":
        df = df[df["product_id"].astype(str) == sel]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Объём отзывов", len(df))
with col2:
    st.metric("Есть темы?", "да" if topic_col in df.columns else "нет")
with col3:
    st.metric("Сентимент обучен?", "да" if sent_available else "нет")

if topic_col in df.columns:
    top_counts = df[topic_col].value_counts(dropna=False).reset_index()
    top_counts.columns = ["topic", "count"]
    fig = px.bar(top_counts, x="topic", y="count", title="Распределение тем")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment prediction (if model exists)
if sent_available:
    with open(sent_model_path, "rb") as f:
        bundle = pickle.load(f)
    clf, le = bundle["clf"], bundle["label_encoder"]
    if "sentiment" in df.columns:
        # Общая гистограмма
        sent_series = df["sentiment"].astype(str)
        fig2 = px.histogram(sent_series, title="Распределение сентимента (все данные)")
        st.plotly_chart(fig2, use_container_width=True)

        # --- Новый блок: распределение по split ---
        if "split" in df.columns:
            split_counts = df.groupby(["split", "sentiment"]).size().reset_index(name="count")
            fig_split = px.bar(
                split_counts,
                x="split",
                y="count",
                color="sentiment",
                barmode="group",
                title="Сентимент по split (train/test)"
            )
            st.plotly_chart(fig_split, use_container_width=True)

            # Табличка с процентами
            st.subheader("Таблица: распределение сентимента по split")
            st.dataframe(
                split_counts.pivot(index="split", columns="sentiment", values="count").fillna(0)
            )
    else:
        st.info("Добавьте колонку 'sentiment' с предсказаниями или обучите модель через `src/sentiment.py`.")

# Topic × Sentiment heatmap (if both exist)
if topic_col in df.columns and "sentiment" in df.columns:
    if method == "lda":
        n_topics = 15
        top_topics = df[topic_col].value_counts().nlargest(n_topics + 1).index
        df_top = df[df[topic_col].isin(top_topics)].copy()
        mapping = {old: i + 1 for i, old in enumerate(sorted(top_topics))}
        df_top["topic_label"] = df_top[topic_col].map(mapping)
    else:
        n_topics = st.slider("Сколько тем оставить в heatmap bertopic?", 5, 465, 15)
        top_topics = df[topic_col].value_counts().nlargest(n_topics + 1).index
        df_top = df[df[topic_col].isin(top_topics)].copy()
        df_top["topic_label"] = df_top[topic_col]

    pivot = df_top.pivot_table(
        index="topic_label",
        columns="sentiment",
        values="text",
        aggfunc="count",
        fill_value=0
    )
    fig3 = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        title=f"Heatmap: Топ-{n_topics} тем × Сентимент"
    ).update_yaxes(tickmode='linear', dtick=1)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# --- Новый блок: ручной ввод отзыва ---
st.markdown("## ✍️ Тест своего отзыва")

user_text = st.text_area("Введите отзыв для анализа", "")

if st.button("Анализировать") and user_text.strip():
    if sent_available:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        emb = encoder.encode([user_text])
        proba = clf.predict_proba(emb)[0]
        pred = le.inverse_transform([clf.predict(emb)[0]])[0]

        st.success(f"Предсказание: **{pred.upper()}**")
        st.write("Вероятности по классам:")
        st.json({cls: float(p) for cls, p in zip(le.classes_, proba)})
    else:
        st.warning("Модель сентимента пока не обучена. Сначала запустите src/sentiment.py.")

st.markdown("---")

st.caption("💡 Подсказка: артефакты создаются скриптами из `src/`. Для BERTopic доступна интерактивная визуализация `bertopic_topics.html`.")
