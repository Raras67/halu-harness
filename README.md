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


#Results (HaluEval, n=100 questions → 200 judgments)Rank
Model
Accuracy
Precision
Recall
F1
Unparseable
1
muse-spark-1.1
0.879
0.963
0.790
0.868
1
2
gemini-3.7-flash
0.874
0.963
0.780
0.862
1
3
grok-4.6
0.873
0.963
0.778
0.860
3
4
claude-sonnet-5
0.870
0.963
0.770
0.856
0

Metric notesPrecision high (~0.96): models rarely flag factual answers as hallucinations.
Recall lower (~0.77–0.79): models miss a non-trivial share of true hallucinations.
F1 spread is small (<0.015): on this slice, models are close; treat ranking as indicative.

Design choicesSame first-N dataset rows for every model (fair head-to-head).
Temperature 0 for deterministic judging.
Parse layer maps free-text YES/NO → {0,1,-1}

