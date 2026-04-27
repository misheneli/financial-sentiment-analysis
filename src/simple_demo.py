import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Простая функция для быстрой проверки
def quick_test():
    # Загрузите ваши данные (или используйте готовую модель если есть)
    df = pd.read_csv('data/processed/financial_news_processed.csv')
    
    # Быстрое обучение модели для демонстрации
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Тестовые примеры
    test_texts = [
        "Company reports strong growth and record profits",
        "Bank faces regulatory issues and declining revenue",
        "The market remained stable with minor fluctuations"
    ]
    
    for text in test_texts:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        print(f"Текст: {text}")
        print(f"Предсказание: {prediction}")
        print("---")

if __name__ == "__main__":
    quick_test()
