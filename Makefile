.PHONY: dev down logs test lint shell redis-cli docker-build

# ── Local Dev ─────────────────────────────────────────────────────

dev:
	docker compose -f docker/docker-compose.yml up --build

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f gateway

shell:
	docker compose -f docker/docker-compose.yml exec gateway bash

redis-cli:
	docker compose -f docker/docker-compose.yml exec redis redis-cli

# ── Testing & Linting ─────────────────────────────────────────────

test:
	python -m pytest tests/ -v

lint:
	ruff check app/ tests/

lint-fix:
	ruff check --fix app/ tests/

# ── Docker ────────────────────────────────────────────────────────

docker-build:
	docker build -f docker/Dockerfile -t llm-gateway:local .

docker-up:
	docker compose -f docker/docker-compose.yml up -d
