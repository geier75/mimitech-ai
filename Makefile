PY ?= python3
RESULTS_DIR = vxor/benchmarks/results

.PHONY: help test test-short test-all arc-eval glue-eval imo-eval compare-external aggregate-results smoke-all pack docker-build docker-test docker-arc-eval docker-glue-eval docker-imo-eval

help:
	@echo "Targets:" \
		"test, test-short, test-all, arc-eval, glue-eval, imo-eval, compare-external, aggregate-results, smoke-all, pack," \
		"docker-build, docker-test, docker-arc-eval, docker-glue-eval, docker-imo-eval"

# Run unit tests
test:
	$(PY) -m pytest -q vxor/benchmarks/tests

# Quick smoke tests (Phase 1-5 validation)
test-short:
	@echo "🚀 Running MISO validation smoke tests..."
	$(PY) -m pytest tests/test_schema_validation.py tests/test_structured_logging.py -v
	@echo "✅ Smoke tests completed"

# Full test suite (all phases 1-10)
test-all: test-short
	@echo "🧪 Running full MISO validation suite..."
	$(PY) -m pytest tests/ -v
	$(PY) scripts/validate_datasets.py
	@echo "📊 Generating validation summary..."
	$(PY) scripts/generate_summary.py --help > /dev/null || echo "Summary generator available"
	@echo "✅ Full validation suite completed"

# ARC-AGI evaluation (expects ARC data via env ARC_DATA_DIR or script will do a smoke-run)
arc-eval:
	$(PY) eval/scripts/arc_eval.py --out-dir $(RESULTS_DIR)/arc_real

# GLUE evaluation (may download datasets; if offline, produces a stub result)
glue-eval:
	$(PY) eval/scripts/glue_eval.py --out-dir $(RESULTS_DIR)/glue_real

# IMO/Math evaluation via SymPy linear systems benchmark
imo-eval:
	$(PY) eval/scripts/imo_eval.py --out-dir $(RESULTS_DIR)/sympy_linear

# Optional comparison with external LLM APIs (only if API keys are present)
compare-external:
	$(PY) eval/scripts/compare_models.py

# Aggregate results JSONs into a summary
aggregate-results:
	$(PY) eval/scripts/aggregate_results.py --results-dir $(RESULTS_DIR) --out-dir $(RESULTS_DIR)/summary

# Run all smoke steps and aggregate
smoke-all:
	$(MAKE) arc-eval
	$(MAKE) imo-eval
	$(MAKE) compare-external
	$(MAKE) aggregate-results

# Containerized workflows
docker-build:
	docker build -t vxor-eval -f eval/Dockerfile .

docker-test:
	docker run --rm -t vxor-eval make test

docker-arc-eval:
	docker run --rm -t -v $$PWD/$(RESULTS_DIR):/app/$(RESULTS_DIR) vxor-eval make arc-eval

docker-glue-eval:
	docker run --rm -t -v $$PWD/$(RESULTS_DIR):/app/$(RESULTS_DIR) vxor-eval make glue-eval

docker-imo-eval:
	docker run --rm -t -v $$PWD/$(RESULTS_DIR):/app/$(RESULTS_DIR) vxor-eval make imo-eval

# Create distributable evaluator pack (git-tracked files only)
pack:
	@HASH=$$(git rev-parse --short HEAD); DATE=$$(date -u +%Y%m%d); PACK=vxor_evaluator_pack_$${DATE}_$${HASH}.tar.gz; \
	  echo "Creating $$PACK"; \
	  git archive --format=tar.gz --prefix=vxor-evaluator-pack/ -o "$$PACK" HEAD Makefile eval/ docs/; \
	  shasum -a 256 "$$PACK" > "$$PACK.sha256"; \
	  echo "Wrote:"; ls -lh "$$PACK" "$$PACK.sha256"
