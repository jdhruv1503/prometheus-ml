setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

demo:
	prometheus init --dataset tests/fixtures/sample.csv --target survived --metric accuracy --budget 10
	prometheus run
