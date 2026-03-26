lint:
	isort .
	ruff check --fix .
	ruff format .