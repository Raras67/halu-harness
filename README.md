# LLM Hallucination Benchmarking Harness

Lightweight, OpenAI-compatible harness for measuring **hallucination detection** performance on [HaluEval](https://github.com/RUCAIBox/HaluEval) (QA split).

## Features
- Router-agnostic client (CleanAPI / OpenRouter / any OpenAI-compatible endpoint)
- HaluEval QA evaluation (right answer vs hallucinated answer)
- Fixed sample prefix for fair multi-model comparison
- Aggregation script → ranking table

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set API_KEY, BASE_URL

#run
python -m src.run_halueval \
  --model "gemini-3.7-flash" \
  --n-samples 100 \
  --out results/halueval_gemini_100.csv


#short analysis
Metric notesPrecision high (~0.96): models rarely flag factual answers as hallucinations.
Recall lower (~0.77–0.79): models miss a non-trivial share of true hallucinations.
F1 spread is small (<0.015): on this slice, models are close; treat ranking as indicative.

Design choicesSame first-N dataset rows for every model (fair head-to-head).
Temperature 0 for deterministic judging.
Parse layer maps free-text YES/NO → {0,1,-1}

