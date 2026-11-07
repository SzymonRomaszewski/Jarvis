.PHONY: test check docs screenshots run clean

test:        ## run all tests
	python -m unittest discover -s tests -v

check:       ## static name check + docs in sync
	python scripts/check_names.py jarvis.py
	python scripts/gen_tools_doc.py --check

docs:        ## regenerate docs/tools.md
	python scripts/gen_tools_doc.py

screenshots: ## regenerate README images (needs: pip install playwright pillow && playwright install chromium)
	python scripts/make_screenshots.py

run:         ## start Jarvis
	python jarvis.py

clean:
	rm -rf build dist *.egg-info __pycache__ tests/__pycache__
