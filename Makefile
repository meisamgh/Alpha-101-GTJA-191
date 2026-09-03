.PHONY: setup lint test validate research trader clean-generated

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev,research]'

lint:
	.venv/bin/ruff check quant_research tests

test:
	.venv/bin/pytest

validate: lint test
	git diff --check

research:
	.venv/bin/python -m quant_research.experiments.local_research --output-dir artifacts
	.venv/bin/python -m quant_research.experiments.supplementary

trader:
	.venv/bin/python -m quant_research.experiments.offline_trader

clean-generated:
	@echo "Generated data and model artifacts are retained intentionally; remove them manually if needed."
