init:
	docker compose up --build -d

up:
	docker compose up -d

stop:
	docker compose stop

reset:
	docker compose down
	make init

format:
	cd backend && uv run ruff check --fix
	cd backend && uv run ruff format

lint:
	cd backend && ruff check
	cd frontend && npm run lint

oapigen:
	cd backend && make oapigen
	cd frontend && npm run orval:backend-api
