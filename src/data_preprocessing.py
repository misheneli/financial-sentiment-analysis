import pandas as pd
import os

def preprocess_financial_news():
    # Загружаем данные
    df = pd.read_csv('data/raw/all-data.csv', encoding='latin-1', header=None)
    
    # Переименовываем колонки
    df.columns = ['sentiment', 'text']
    
    # Очищаем текст (убираем лишние пробелы)
    df['text'] = df['text'].str.strip()
    
    # Создаем папку processed если ее нет
    os.makedirs('data/processed', exist_ok=True)
    
    # Сохраняем обработанные данные
    processed_path = 'data/processed/financial_news_processed.csv'
    df.to_csv(processed_path, index=False)
    
    print(f"Обработанные данные сохранены в: {processed_path}")
    print(f"Размер датасета: {len(df)} записей")
    print("Распределение sentiment:")
    print(df['sentiment'].value_counts())
    
    return df

if __name__ == "__main__":
    preprocess_financial_news()
