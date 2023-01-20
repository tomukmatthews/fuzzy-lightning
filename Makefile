.PHONY: lint test build upload clean

format:
	black fuzzy_lightning tests
	isort fuzzy_lightning tests

lint:
	flake8 fuzzy_lightning tests

test:
	pytest tests

build:
	python setup.py sdist bdist_wheel

upload:
	twine upload dist/*

clean:
	rm -rf build dist .eggs *.egg-info venv