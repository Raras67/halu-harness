# Methodology — HaluEval Hallucination Detection

## Dataset
- Source: `pminervini/HaluEval`, config `qa`, split `data`
- Each item: `knowledge`, `question`, `right_answer`, `hallucinated_answer`
- Evaluation uses the **first N rows** (fixed prefix) so all models see the same items

## Protocol
1. For each question, run the model **twice** as a binary judge:
   - once on `right_answer` (ground truth label = 0, factual)
   - once on `hallucinated_answer` (ground truth label = 1)
2. Prompt asks whether the response contains hallucination given knowledge + question
3. Temperature = 0; short `max_tokens`
4. Parse free-text to `{1=YES, 0=NO, -1=unparseable}`

## Metrics
| Metric | Definition |
|--------|------------|
| Accuracy | Correct labels / valid predictions |
| Precision | TP / (TP+FP) — of predicted hallucinations, how many are real |
| Recall | TP / (TP+FN) — of real hallucinations, how many caught |
| F1 | Harmonic mean of precision and recall |
| Yes-ratio | Share of predictions equal to 1 |
| Unparseable | Count of parse failures (excluded from classification metrics) |

## Limitations
- N=100 questions (200 judgments) — ranking gaps may not be statistically large
- Judge is the same model under test (detection ability, not answer generation)
- English QA slice only; prompt-sensitive
