.PHONY: lint test

format:
	black fuzzy_lightning tests
	isort fuzzy_lightning tests

lint:
	flake8 fuzzy_lightning tests

test:
	pytest tests