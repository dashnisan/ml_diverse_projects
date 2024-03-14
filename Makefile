install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval-lax pca_autoencoder_morphometrics.ipynb

format:
	black *.py

lint:
	pylint --disable=R,C *.py

all: install lint test