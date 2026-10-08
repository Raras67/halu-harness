# LLM Hallucination Benchmarking Harness

Lightweight, OpenAI-compatible harness for measuring **hallucination detection** performance on [HaluEval](https://github.com/RUCAIBox/HaluEval) (QA split).

## Features
- Router-agnostic client (CleanAPI / OpenRouter / any OpenAI-compatible endpoint)
- HaluEval QA evaluation (right answer vs hallucinated answer)
- Fixed sample prefix for fair multi-model comparison
- Aggregation script → ranking table
- Metrics: accuracy, precision, recall, F1, yes-ratio
- Unit tests for YES/NO parsing

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set API_KEY, BASE_URL

## run
python -m src.run_halueval \
  --model "gemini-3.7-flash" \
  --n-samples 100 \
  --out results/halueval_gemini_100.csv

PYTHONPATH=. python scripts/compare_results.py
PYTHONPATH=. pytest -q

#short analysis
Metric notesPrecision high (~0.96): models rarely flag factual answers as hallucinations.
Recall lower (~0.77–0.79): models miss a non-trivial share of true hallucinations.
F1 spread is small (<0.015): on this slice, models are close; treat ranking as indicative.

Design choicesSame first-N dataset rows for every model (fair head-to-head).
Temperature 0 for deterministic judging.
Parse layer maps free-text YES/NO → {0,1,-1}

## results
Findings-All four models show high precision (~0.96) and lower recall (~0.77–0.79): they rarely false-flag factual answers, but miss a non-trivial fraction of true hallucinations. F1 spread is < 0.015 on this slice—treat ranking as indicative, not a large capability gap.Failing / hard cases (qualitative)Typical failure modes observed in unparseable or wrong rows:Under-detection (false negative): fluent hallucinated answers that stay close to the knowledge wording.
Parse fragility: answers that hedge (“possibly yes”) instead of YES/NO.
Knowledge conflict: long knowledge fields with two entities; model confuses which fact applies.

See docs/methodology.md for full protocol.Design choicesSame first-N dataset rows for every model
Temperature 0
Parse layer: free-text → {0, 1, -1} 

