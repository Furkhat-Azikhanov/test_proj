
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Подгружаем spaCy модель для обработки русского языка
nlp = spacy.load("ru_core_news_sm")

# Заранее подготовленный набор текстов, по которым будем искать
data = [
    "Это пример текста о кошках.",
    "Собаки - лучшие друзья человека.",
    "Программирование на Python очень популярно.",
    "Книги об искусственном интеллекте становятся всё более популярными.",
    "Кошки любят спать целыми днями."
]

# Простой пайплайн обработки текста: убираем лишнее и оставляем только полезные слова
def nlp_pipeline(text):
    """
    Превращает текст в список "чистых" лемм.
    Убираем стоп-слова и знаки препинания.
    """
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# TF-IDF используется для преобразования текстов в числовые вектора
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
tfidf_matrix = tfidf_vectorizer.fit_transform([nlp_pipeline(text) for text in data])

# Функция поиска по базе
def search(query, top_n=3):
    """
    Ищем наиболее релевантные тексты для переданного запроса.
    Возвращаем топ-N результатов.
    """
    query_tokens = nlp_pipeline(query)
    query_vector = tfidf_vectorizer.transform([query_tokens])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0, -top_n:][::-1]
    return [data[i] for i in top_indices]

# Flask-приложение для работы с API
app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_text():
    """
    Обрабатываем текст пользователя через наш NLP пайплайн.
    Возвращаем список лемматизированных слов.
    """
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "Текст запроса не может быть пустым"}), 400
    tokens = nlp_pipeline(input_text)
    return jsonify({"processed_text": tokens})

@app.route("/search", methods=["POST"])
def search_text():
    """
    Поиск по текстам в базе. 
    Пользователь передает запрос, а мы ищем 3 наиболее релевантных текста.
    """
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Текст запроса не может быть пустым"}), 400
    results = search(query)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
