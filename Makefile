install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval-lax ./pca_egyptian_mummies/*.ipynb

format:
	black *.py

lint:
	pylint --disable=R,C *.py

all: install lint test