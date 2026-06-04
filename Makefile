# Convenience targets. Run `make help` for the list.
.PHONY: help install dev test lint run notebook clean

help:
	@echo "install   - install runtime deps"
	@echo "dev       - editable install with dev + notebook extras"
	@echo "test      - run offline test suite (no API key needed)"
	@echo "lint      - ruff lint"
	@echo "run       - interactive CLI (role=analyst)"
	@echo "clean     - remove caches and generated data"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev,notebook]"

test:
	pytest

lint:
	ruff check .

run:
	python -m classified_doc_poc --role analyst

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ data *.db *.faiss
