.PHONY: install test train evaluate lint api dashboard

# ── Setup ──
install:
	pip install -r requirements.txt

# ── Testing ──
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=credicouncil --cov-report=html

# ── ML Pipeline ──
train:
	python training/train.py --data-dir home-credit-default-risk/

evaluate:
	python training/evaluate.py --data-dir home-credit-default-risk/ --model-path models/lgbm_v1.pkl

pilot:
	python training/pilot_test.py --cases 40

ablation:
	python training/evaluate.py --ablation --data-dir home-credit-default-risk/

# ── API ──
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Dashboard ──
dashboard:
	cd dashboard && npm run dev

# ── Code Quality ──
lint:
	ruff check credicouncil/ api/ training/ tests/

format:
	black credicouncil/ api/ training/ tests/
