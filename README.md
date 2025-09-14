
# Анализ пользовательских отзывов с NLP

## Быстрый старт

```bash
python -m venv venv
pip install -r requirements.txt
```


1) Загрузка IMDB датасета

```bash
python src/download_imdb.py
```

2) Запустите офлайн обработку (создаст артефакты в `data/artifacts/`):
```bash
python src/preprocess.py --input data/reviews.csv --lang en
python src/topic_modeling.py --input data/clean_reviews.parquet --method lda --num-topics 15
python src/topic_modeling.py --input data/clean_reviews.parquet --method bertopic
python src/sentiment.py --input data/clean_reviews.parquet 
```

3) Запустите дашборд:
```bash
streamlit run app/streamlit_app.py
```

## Структура
- `src/preprocess.py` — чистка, токенизация, лемматизация/стемминг, сохранение parquet.
- `src/topic_modeling.py` — LDA на gensim и BERTopic; сохраняет темы, присвоения тем и интерактивы (<a href="http://localhost:63342/review-nlp/data/artifacts/lda_vis.html#topic=10&lambda=0.19&term=" target="_blank" rel="noopener noreferrer">
  Визуализация LDA
</a> и <a href="http://localhost:63342/review-nlp/data/artifacts/bertopic_topics.html" target="_blank" rel="noopener noreferrer">
  Визуализация BERTopic
</a>).
- `src/sentiment.py` — обучение сентимента: Logistic Regression на эмбеддингах `sentence-transformers` (быстро) или fine-tune DistilBERT (флаг `--finetune`).
- `app/streamlit_app.py` — <a href="https://reviewanalysis-hfbeb4tx9jmuehonf4tgu9.streamlit.app/" target="_blank" rel="noopener noreferrer">
  Интерактивный дашборд
</a>.
- `report.md` — [отчёт.](report.md)

