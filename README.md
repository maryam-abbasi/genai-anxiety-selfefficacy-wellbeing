# genai-anxiety-selfefficacy-wellbeing

# Generative AI Use, Academic Anxiety and Student Well-Being

## Contents

- `dataset.csv` — anonymised survey data, N = 627, 47 columns. One row per
  participant, identified only by an anonymous ID (`R0001`-`R0627`).
- `02_Codebook.docx` — full variable list: descriptions, item wordings for
  all scales, coding, and composite-score formulas.
- `analysis_script.py` — reproduces the descriptive statistics, hierarchical
  regression, parallel mediation model, and the exploratory/confirmatory
  factor analysis of the AI-related academic anxiety scale reported in the
  manuscript.
- `requirements.txt` — package versions needed to run the script.

## Reproducing the analysis

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python analysis_script.py
```

The script expects `dataset.csv` in the same directory and writes its
output (tables and statistics matching those in the manuscript) to the
console.

## Data collection

Anonymous online survey administered February-April 2025 at four universities in Portugal. 
Participation was voluntary and self-selected; see the manuscript's Methods and Limitations
sections for details on recruitment and the resulting constraints on inference.

## Measures

Full item wordings and scoring for all instruments (AI-related academic
anxiety, academic self-efficacy, UWES-S-9 engagement, WHO-5 well-being) are
in the codebook. Composite scores in `dataset.csv` are pre-computed
following the formulas listed there.

## Citation

Citation details will be added once the manuscript has completed review
and the author-withheld version of this repository is replaced with the
attributed one.
