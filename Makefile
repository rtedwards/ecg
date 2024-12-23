NC='\033[0m'
YELLOW='\x1b[38;5;220m'

.PHONY: format lint static-analysis test ci-checks

format:
	@echo
	@echo -e ${YELLOW} ---- Formatting ---- ${NC}
	ruff format --check

format-fix:
	@echo
	@echo -e ${YELLOW} ---- Formatting ---- ${NC}
	ruff format

lint:
	@echo 
	@echo -e ${YELLOW} ---- Linting ---- ${NC}
	ruff check ecg


lint-fix:
	@echo 
	@echo -e ${YELLOW} ---- Linting ---- ${NC}
	ruff check ecg --fix


static-analysis:
	@echo 
	@echo -e ${YELLOW} ---- Static Analysis ---- ${NC}
	mypy ecg/*.py --follow-import=skip

tests:
	@echo
	@echo -e ${YELLOW} ---- Tests ---- ${NC}
	pytest .

test-coverage:
	@echo 
	@echo -e ${YELLOW} ---- Test Coverage ---- ${NC}
	pytest . --cov='.' --cov-report=term --cov-report xml:coverage.xml


# Fix formatting and linting where possible
fix: format-fix lint-fix 

# Run the same checks as in CI
ci-checks: format lint static-analysis tests

colors:
	@echo -e ${YELLOW} ---- Formatting ---- ${NC}
