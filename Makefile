.PHONY: install preprocess train predict backtest test clean

install:
	pip install -r requirements.txt

preprocess:
	python src/data_preprocessing.py

train:
	python src/train_model.py

predict:
	python src/predict.py

backtest:
	python src/backtester.py

test:
	pytest tests/ -v --tb=short

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f analysis_results/*.png analysis_results/*.csv analysis_results/*.txt

all: preprocess train backtest
