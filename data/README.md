# Data

The real-text experiments use three public-domain books from NLTK's
Gutenberg corpus, downloaded automatically on first run:

- `austen-emma.txt` (Jane Austen, *Emma*)
- `melville-moby_dick.txt` (Herman Melville, *Moby Dick*)
- `chesterton-thursday.txt` (G. K. Chesterton, *The Man Who Was Thursday*)

No data files are committed; `experiments/run_corpora.py` fetches them
via `nltk.download("gutenberg")`.
