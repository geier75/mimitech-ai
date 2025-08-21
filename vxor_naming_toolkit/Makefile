# VXOR Case-Consistency Toolkit - Nano-Step TDD Makefile
# =====================================================
# 
# Automated workflow for nano-step TDD approach:
# 1. scan -> 2. test (fail) -> 3. fix -> 4. test (pass) -> 5. commit -> 6. report

SHELL := /bin/bash
PYTHON := python3
CLI := $(PYTHON) vxor_naming_nano_cli.py
REPO_ROOT := /Volumes/My Book/MISO_Ultimate 15.32.28
REPORTS_DIR := reports
MAX_LOOPS := 100
FAIL_FAST := true

# Category processing order
CATEGORIES := directories files imports symbols shims docs_cli
CURRENT_CATEGORY ?= directories

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NC := \033[0m # No Color

.PHONY: help init scan fix-one verify test report status clean all-categories
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "🧪 VXOR Case-Consistency Toolkit - Nano-Step TDD"
	@echo "================================================"
	@echo ""
	@echo "Usage: make <target> [CURRENT_CATEGORY=<category>]"
	@echo ""
	@echo "Main Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Categories: $(CATEGORIES)"
	@echo "Current Category: $(CURRENT_CATEGORY)"

init: ## Initialize toolkit (run once)
	@echo "🔧 $(YELLOW)Initializing VXOR Case-Consistency Toolkit$(NC)"
	@mkdir -p $(REPORTS_DIR)
	@touch $(REPORTS_DIR)/nano_steps.jsonl
	@touch $(REPORTS_DIR)/violations.json
	@echo "✅ $(GREEN)Toolkit initialized$(NC)"

scan: ## Scan for violations in current category
	@echo "🔍 $(BLUE)Scanning category: $(CURRENT_CATEGORY)$(NC)"
	@$(CLI) scan --category $(CURRENT_CATEGORY) --limit 10 --out $(REPORTS_DIR)

scan-single: ## Scan for single violation (nano-step)
	@echo "🔍 $(BLUE)Nano-step scan: $(CURRENT_CATEGORY)$(NC)"
	@$(CLI) scan --category $(CURRENT_CATEGORY) --limit 1 --out $(REPORTS_DIR)

fix-one: scan-single ## Fix exactly one violation (nano-step TDD loop)
	@echo "🧪 $(YELLOW)Starting nano-step TDD loop for $(CURRENT_CATEGORY)$(NC)"
	@echo "1️⃣ Getting next violation..."
	@$(eval VIOLATION_ID := $(shell $(CLI) scan --category $(CURRENT_CATEGORY) --limit 1 --output-format json | jq -r '.[0].id // empty' 2>/dev/null))
	@if [ -z "$(VIOLATION_ID)" ]; then \
		echo "✅ $(GREEN)No violations found in category $(CURRENT_CATEGORY)$(NC)"; \
		exit 0; \
	fi
	@echo "🎯 $(YELLOW)Processing violation: $(VIOLATION_ID)$(NC)"
	@echo ""
	@echo "2️⃣ Running tests (should FAIL)..."
	@$(MAKE) test-violation VIOLATION_ID=$(VIOLATION_ID) || echo "✅ $(GREEN)Test failed as expected (TDD)$(NC)"
	@echo ""
	@echo "3️⃣ Showing fix preview..."
	@$(CLI) fix --category $(CURRENT_CATEGORY) --id $(VIOLATION_ID) --dry-run
	@echo ""
	@read -p "Apply this fix? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "4️⃣ Applying fix..."; \
		$(CLI) fix --category $(CURRENT_CATEGORY) --id $(VIOLATION_ID) --apply; \
		echo ""; \
		echo "5️⃣ Running tests (should PASS)..."; \
		$(MAKE) test-violation VIOLATION_ID=$(VIOLATION_ID) && echo "✅ $(GREEN)Test passed after fix$(NC)" || (echo "❌ $(RED)Test still failing after fix$(NC)" && exit 1); \
		echo ""; \
		echo "6️⃣ Verifying system state..."; \
		$(MAKE) verify; \
		echo ""; \
		echo "7️⃣ Committing changes..."; \
		$(MAKE) commit-nano-step VIOLATION_ID=$(VIOLATION_ID) CATEGORY=$(CURRENT_CATEGORY); \
		echo ""; \
		echo "8️⃣ Updating reports..."; \
		$(CLI) report --append; \
		echo ""; \
		echo "🎉 $(GREEN)Nano-step completed successfully!$(NC)"; \
	else \
		echo "❌ $(RED)Fix cancelled by user$(NC)"; \
		exit 1; \
	fi

test: ## Run full test suite
	@echo "🧪 $(BLUE)Running test suite$(NC)"
	@$(PYTHON) -m pytest test_nano_steps.py -v --tb=short

test-violation: ## Test specific violation (requires VIOLATION_ID)
	@echo "🧪 $(BLUE)Testing violation: $(VIOLATION_ID)$(NC)"
	@$(PYTHON) -m pytest test_nano_steps.py::test_violation_by_id -v --tb=short -k $(VIOLATION_ID) || exit 1

verify: ## Verify system state (imports, compliance, tests)
	@echo "✅ $(BLUE)Verifying system state$(NC)"
	@$(CLI) verify

report: ## Generate comprehensive reports
	@echo "📊 $(BLUE)Generating reports$(NC)"
	@$(CLI) report

status: ## Show current status of all categories
	@echo "📊 $(BLUE)VXOR Case-Consistency Status$(NC)"
	@echo "================================="
	@for category in $(CATEGORIES); do \
		echo ""; \
		echo "Category: $$category"; \
		violations=$$($(CLI) scan --category $$category --output-format json 2>/dev/null | jq length 2>/dev/null || echo "0"); \
		if [ "$$violations" -eq 0 ]; then \
			echo "  ✅ $(GREEN)Clean (0 violations)$(NC)"; \
		else \
			echo "  ⚠️  $(YELLOW)$$violations violations pending$(NC)"; \
		fi; \
	done

commit-nano-step: ## Commit nano-step fix (requires VIOLATION_ID and CATEGORY)
	@echo "📝 $(BLUE)Committing nano-step fix$(NC)"
	@cd $(REPO_ROOT) && git add -A
	@cd $(REPO_ROOT) && git commit -m "chore(naming:$(CATEGORY)): fix $(VIOLATION_ID) [nano-step]" || echo "⚠️  $(YELLOW)Nothing to commit$(NC)"

# Category-specific targets
fix-directories: ## Fix all directory violations
	@$(MAKE) fix-category-loop CURRENT_CATEGORY=directories

fix-files: ## Fix all file violations  
	@$(MAKE) fix-category-loop CURRENT_CATEGORY=files

fix-imports: ## Fix all import violations
	@$(MAKE) fix-category-loop CURRENT_CATEGORY=imports

fix-symbols: ## Fix all symbol violations
	@$(MAKE) fix-category-loop CURRENT_CATEGORY=symbols

fix-shims: ## Fix all shim violations
	@$(MAKE) fix-category-loop CURRENT_CATEGORY=shims

fix-docs: ## Fix all documentation violations
	@$(MAKE) fix-category-loop CURRENT_CATEGORY=docs_cli

fix-category-loop: ## Internal: Loop through all violations in category
	@echo "🔄 $(YELLOW)Processing all violations in category: $(CURRENT_CATEGORY)$(NC)"
	@count=0; \
	while [ $$count -lt $(MAX_LOOPS) ]; do \
		violations=$$($(CLI) scan --category $(CURRENT_CATEGORY) --output-format json 2>/dev/null | jq length 2>/dev/null || echo "0"); \
		if [ "$$violations" -eq 0 ]; then \
			echo "✅ $(GREEN)Category $(CURRENT_CATEGORY) completed (0 violations)$(NC)"; \
			break; \
		fi; \
		echo ""; \
		echo "🔄 $(YELLOW)Loop $$count: $$violations violations remaining$(NC)"; \
		$(MAKE) fix-one CURRENT_CATEGORY=$(CURRENT_CATEGORY) || (echo "❌ $(RED)Fix failed$(NC)" && exit 1); \
		count=$$((count + 1)); \
	done

all-categories: ## Process all categories in order
	@echo "🚀 $(YELLOW)Processing all categories in order$(NC)"
	@for category in $(CATEGORIES); do \
		echo ""; \
		echo "========================================"; \
		echo "🔄 $(BLUE)Starting category: $$category$(NC)"; \
		echo "========================================"; \
		$(MAKE) fix-category-loop CURRENT_CATEGORY=$$category || (echo "❌ $(RED)Failed on category: $$category$(NC)" && exit 1); \
		echo "✅ $(GREEN)Completed category: $$category$(NC)"; \
	done
	@echo ""
	@echo "🎉 $(GREEN)All categories processed successfully!$(NC)"

# CI/Development targets
check-dependencies: ## Check required dependencies
	@echo "🔍 $(BLUE)Checking dependencies$(NC)"
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "❌ $(RED)Python 3 not found$(NC)" && exit 1)
	@command -v jq >/dev/null 2>&1 || (echo "❌ $(RED)jq not found (install with: brew install jq)$(NC)" && exit 1)
	@command -v git >/dev/null 2>&1 || (echo "❌ $(RED)git not found$(NC)" && exit 1)
	@$(PYTHON) -c "import pytest" 2>/dev/null || (echo "❌ $(RED)pytest not available$(NC)" && exit 1)
	@echo "✅ $(GREEN)All dependencies available$(NC)"

install-pre-commit: ## Install pre-commit hooks
	@echo "🔧 $(BLUE)Installing pre-commit hooks$(NC)"
	@cd $(REPO_ROOT) && cp $(PWD)/.pre-commit-config.yaml .
	@cd $(REPO_ROOT) && pre-commit install || echo "⚠️  $(YELLOW)pre-commit not available$(NC)"

ci-setup: ## Setup for CI environment
	@echo "🔧 $(BLUE)Setting up CI environment$(NC)"
	@$(MAKE) check-dependencies
	@$(MAKE) init
	@echo "✅ $(GREEN)CI setup completed$(NC)"

ci-test: ## Run CI test sequence
	@echo "🧪 $(BLUE)Running CI test sequence$(NC)"
	@$(MAKE) scan
	@$(MAKE) test
	@$(MAKE) verify
	@echo "✅ $(GREEN)CI tests passed$(NC)"

# Utility targets
clean: ## Clean generated files
	@echo "🧹 $(BLUE)Cleaning generated files$(NC)"
	@rm -rf $(REPORTS_DIR)/*.json $(REPORTS_DIR)/*.jsonl $(REPORTS_DIR)/*.md
	@rm -rf __pycache__ .pytest_cache
	@echo "✅ $(GREEN)Clean completed$(NC)"

debug-violation: ## Debug specific violation (requires VIOLATION_ID)
	@echo "🐛 $(BLUE)Debugging violation: $(VIOLATION_ID)$(NC)"
	@$(CLI) scan --output-format json | jq '.[] | select(.id=="$(VIOLATION_ID)")'

show-categories: ## Show all available categories
	@echo "📋 $(BLUE)Available Categories:$(NC)"
	@for category in $(CATEGORIES); do \
		echo "  - $$category"; \
	done

# Quick start targets
quick-start: ## Quick start: scan all and show status
	@echo "🚀 $(YELLOW)VXOR Case-Consistency Quick Start$(NC)"
	@$(MAKE) init
	@$(MAKE) status
	@echo ""
	@echo "Next steps:"
	@echo "  1. $(BLUE)make fix-directories$(NC) - Fix directory naming"
	@echo "  2. $(BLUE)make fix-files$(NC) - Fix file naming"  
	@echo "  3. $(BLUE)make fix-imports$(NC) - Fix import statements"
	@echo "  4. $(BLUE)make all-categories$(NC) - Process all categories"

demo: ## Run demo of nano-step process
	@echo "🎬 $(YELLOW)Demo: Nano-Step TDD Process$(NC)"
	@echo "================================="
	@echo ""
	@echo "1. Scan for violations:"
	@$(MAKE) scan-single CURRENT_CATEGORY=directories || true
	@echo ""
	@echo "2. This is how you would fix one violation:"
	@echo "   $(BLUE)make fix-one CURRENT_CATEGORY=directories$(NC)"
	@echo ""
	@echo "3. Process all violations in a category:"
	@echo "   $(BLUE)make fix-directories$(NC)"
	@echo ""
	@echo "4. Process all categories:"
	@echo "   $(BLUE)make all-categories$(NC)"

# Help for specific workflows
help-nano-step: ## Show nano-step workflow help
	@echo "🧪 $(YELLOW)Nano-Step TDD Workflow$(NC)"
	@echo "========================="
	@echo ""
	@echo "The nano-step approach fixes exactly one violation per loop:"
	@echo ""
	@echo "1. $(BLUE)make scan-single$(NC) - Find one violation"
	@echo "2. $(BLUE)make test-violation VIOLATION_ID=<id>$(NC) - Test (should fail)"
	@echo "3. $(BLUE)make fix-one$(NC) - Apply minimal fix"
	@echo "4. $(BLUE)make test-violation VIOLATION_ID=<id>$(NC) - Test (should pass)"
	@echo "5. $(BLUE)make commit-nano-step$(NC) - Commit changes"
	@echo ""
	@echo "Automated: $(BLUE)make fix-one$(NC) does steps 1-5"

help-categories: ## Show category processing help
	@echo "📂 $(YELLOW)Category Processing Order$(NC)"
	@echo "========================="
	@echo ""
	@echo "Categories are processed in this order:"
	@for i in $$(seq 1 $$(echo $(CATEGORIES) | wc -w)); do \
		category=$$(echo $(CATEGORIES) | cut -d' ' -f$$i); \
		echo "$$i. $(BLUE)$$category$(NC)"; \
	done
	@echo ""
	@echo "Process single category: $(BLUE)make fix-<category>$(NC)"
	@echo "Process all categories:  $(BLUE)make all-categories$(NC)"
