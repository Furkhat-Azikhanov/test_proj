
# Тестовое задание: NLP-пайплайн и поиск

## Описание
Данный проект реализует базовый NLP-пайплайн и REST API для обработки текстовых данных на русском языке.

### Возможности:
1. Обработка текста: токенизация, очистка от стоп-слов и лемматизация.
2. Поиск по базе текстов с использованием TF-IDF и cosine similarity.
3. REST API для обработки и поиска.

## Установка
1. Убедитесь, что Python версии 3.11 установлен.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Скачайте модель для spaCy:
   ```bash
   python -m spacy download ru_core_news_sm
   ```

## Запуск
1. Запустите сервер:
   ```bash
   python app.py
   ```
2. Используйте API для обработки и поиска текста.

### Примеры запросов:
**Обработка текста:**
```bash
curl -X POST http://127.0.0.1:5000/process -H "Content-Type: application/json" -d '{"text": "Программирование очень важно."}'
```
**Поиск текста:**
```bash
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{"query": "Python программирование"}'
```

## Зависимости
- Flask
- scikit-learn
- spaCy
