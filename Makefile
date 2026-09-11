PYTHON ?= python3

.PHONY: test reproduce baselines rescore

test:
	$(PYTHON) -m pytest

reproduce: test
	$(PYTHON) tools/paper_tables.py
	$(PYTHON) tools/figures.py
	$(PYTHON) tools/prompt_appendix.py

baselines:
	mkdir -p reproduced
	$(PYTHON) run_v0.py --universes 100 --out reproduced/baselines.json
	$(PYTHON) run_twins.py --seeds 60 --out reproduced/twins.json

rescore:
	$(PYTHON) tools/rescore_review.py
