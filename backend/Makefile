EXEC := docker compose exec api uv run

.PHONY: lint
lint:
	$(EXEC) ruff format --check
	$(EXEC) ruff check

.PHONY: format
format:
	$(EXEC) ruff format
	$(EXEC) ruff check --select I --fix

.PHONY: oapigen
oapigen:
	$(EXEC) python -m api.commands.create_openapi_schema
