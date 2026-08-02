.PHONY: pdf quick clean

DOC = reinforce-to-grpo

pdf: $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex   # second pass resolves \cref and the ToC

quick: $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).fls $(DOC).fdb_latexmk
