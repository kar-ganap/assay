.PHONY: pdf quick verify clean

DOC = reinforce-to-grpo

# `verify` is part of `pdf`, not an optional extra. A cross-reference whose *name* fails to resolve
# (cleveref with an unregistered environment) prints "??" in the output while logging nothing at all,
# so log-grepping cannot catch it — the rendered text is the only reliable check.
pdf: $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex   # resolves \cref and the ToC
	pdflatex -interaction=nonstopmode $(DOC).tex   # settles ToC page numbers after reflow
	@$(MAKE) --no-print-directory verify

quick: $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex

verify: $(DOC).pdf
	@fail=0; \
	if pdftotext $(DOC).pdf - 2>/dev/null | grep -q '??'; then \
	  echo "FAIL: '??' in rendered output — unresolved cross-reference name:"; \
	  pdftotext $(DOC).pdf - 2>/dev/null | grep -n '??' | head -20; fail=1; \
	fi; \
	if grep -q 'undefined' $(DOC).log 2>/dev/null; then \
	  echo "FAIL: undefined reference in $(DOC).log:"; \
	  grep -n 'undefined' $(DOC).log | head -10; fail=1; \
	fi; \
	if grep -qE '^!' $(DOC).log 2>/dev/null; then \
	  echo "FAIL: LaTeX error:"; grep -nE '^!' $(DOC).log | head -10; fail=1; \
	fi; \
	if [ $$fail -eq 0 ]; then \
	  echo "OK: $$(pdfinfo $(DOC).pdf | awk '/^Pages/{print $$2}') pages, no unresolved refs"; \
	fi; \
	exit $$fail

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).fls $(DOC).fdb_latexmk
