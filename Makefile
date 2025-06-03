init:
	docker compose up --build -d

up:
	docker compose up -d

stop:
	docker compose stop

reset:
	docker compose down
	make init
