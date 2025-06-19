# ======== Project Setup ========

.PHONY: install clean lint format test run

install:
	poetry install
	pre-commit install

setup:
	poetry shell

# ======== Development ========

lint:
	poetry run ruff bias_audit_tool
	poetry run black --check bias_audit_tool

format:
	poetry run black bias_audit_tool
	poetry run ruff bias_audit_tool --fix

test:
	poetry run pytest tests

# ======== App Run ========

run:
	poetry run streamlit run bias_audit_tool/app.py

# ======== Maintenance ========

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

lock:
	poetry lock

update:
	poetry update

check:
	poetry check

precommit:
	pre-commit run --all-files
