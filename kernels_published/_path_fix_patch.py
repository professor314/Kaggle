import re, sys, pathlib
HELPER = '''
def _find_input(slug):
    """Locate a competition's input dir across Kaggle mount layouts."""
    import os, glob
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{slug} not found. Inputs: {glob.glob('/kaggle/input/*')}")

'''
def patch(path, subs):
    p = pathlib.Path(path); s = p.read_text(encoding="utf-8")
    if "_find_input" in s: return
    for old, new in subs: 
        assert old in s, (path, old); s = s.replace(old, new)
    # insert helper after the module docstring / first import block start
    m = re.search(r'^(import |from )', s, re.M)
    s = s[:m.start()] + HELPER.lstrip("\n") + "\n" + s[m.start():]
    p.write_text(s, encoding="utf-8")
patch("digit-recognizer-5-fold-cnn-ensemble-tta/digit-recognizer-5-fold-cnn-ensemble-tta.py", [('Path("/kaggle/input/digit-recognizer")','Path(_find_input("digit-recognizer"))')])
patch("house-prices-boosting-linear-oof-stack/house-prices-boosting-linear-oof-stack.py", [('Path("/kaggle/input/house-prices-advanced-regression-techniques")','Path(_find_input("house-prices-advanced-regression-techniques"))')])
patch("home-data-ml-boosting-linear-oof-stack/home-data-ml-boosting-linear-oof-stack.py", [('Path("/kaggle/input/home-data-for-ml-course")','Path(_find_input("home-data-for-ml-course"))')])
patch("store-sales-per-family-lightgbm/store-sales-per-family-lightgbm.py", [('Path("/kaggle/input/store-sales-time-series-forecasting")','Path(_find_input("store-sales-time-series-forecasting"))')])
patch("nlp-disaster-tweets-3way-ensemble/nlp-disaster-tweets-3way-ensemble.py", [('Path("/kaggle/input/nlp-getting-started")','Path(_find_input("nlp-getting-started"))')])
patch("tpu-getting-started-baseline/tpu-getting-started-baseline.py", [('"/kaggle/input/tpu-getting-started"','_find_input("tpu-getting-started")')])
patch("llm-finetuning-tfidf-baseline/llm-finetuning-tfidf-baseline.py", [("'/kaggle/input/llm-classification-finetuning/'","_find_input('llm-classification-finetuning') + '/'")])
patch("contradictory-watson-tfidf/contradictory-watson-tfidf.py", [("'/kaggle/input/contradictory-my-dear-watson/train.csv'","_find_input('contradictory-my-dear-watson') + '/train.csv'"),("'/kaggle/input/contradictory-my-dear-watson/test.csv'","_find_input('contradictory-my-dear-watson') + '/test.csv'")])
print("patched")
