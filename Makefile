ENV ?= .venv

install:
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

train:
	uv run python -m moneyscam.pipeline.train_classical --config configs/train.yaml

api:
	uv run uvicorn moneyscam.serving.api:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	uv run streamlit run ui/streamlit_app.py

lint:
	uv run ruff check .
	uv run black --check .

test:
	uv run pytest -q
