import joblib
import pandas as pd

# Загрузка модели и векторизатора
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def demo():
    print("Демонстрация работы модели анализа sentiment финансовых новостей")
    print("Введите 'quit' для выхода")
    print("-" * 50)
    
    while True:
        text = input("\nВведите финансовый текст для анализа: ")
        
        if text.lower() == 'quit':
            break
            
        # Анализируем текст
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        
        # Выводим результат
        print(f"\nРезультат анализа:")
        print(f"Предсказание: {prediction}")
        print("Вероятности:")
        print(f"  Negative: {probability[0]:.2%}")
        print(f"  Neutral: {probability[1]:.2%}")
        print(f"  Positive: {probability[2]:.2%}")
        
        # Интерпретация результата
        if prediction == 'positive':
            print("✅ Положительный sentiment - хорошие новости для инвесторов")
        elif prediction == 'negative':
            print("❌ Отрицательный sentiment - потенциальные риски")
        else:
            print("➖ Нейтральный sentiment - ситуация стабильна")

if __name__ == "__main__":
    demo()
