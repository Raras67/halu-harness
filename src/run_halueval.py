"""
Run HaluEval QA evaluation using an OpenAI-compatible router.

Usage:
    python -m src.run_halueval \
        --model "google/gemini-3.7-flash" \
        --n-samples 200 \
        --out results/halueval_gemini.csv
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from src.config import ENV_PATH
from dotenv import load_dotenv
load_dotenv(dotenv_path=ENV_PATH, override=False)
from tqdm import tqdm

from src.datasets import load_halueval
from src.prompts import HALUEVAL_JUDGE_PROMPT
from src.clients import get_client



def parse_yes_no(text: str) -> int:
    """Extract 0/1 from model response. Returns -1 if unparseable."""
    t = text.strip().upper()
    if t.startswith("YES"):
        return 1
    if t.startswith("NO"):
        return 0
    has_yes = "YES" in t
    has_no = "NO" in t
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return -1


def evaluate(
    model: str,
    n_samples: int,
    out_path: str,
    config: str = "qa",
    max_tokens: int = 8,
    sleep_between: float = 0.0,
):
    ds = load_halueval(config=config, max_samples=n_samples)
    client = get_client()

    rows = []
    for i, row in enumerate(tqdm(ds, desc=f"halueval::{model}")):
        # Test BOTH the right answer and the hallucinated answer
        for label, response in [(0, row["right_answer"]),
                                (1, row["hallucinated_answer"])]:
            prompt = HALUEVAL_JUDGE_PROMPT.format(
                knowledge=row["knowledge"],
                question=row["question"],
                response=response,
            )
            try:
                raw = client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                pred = parse_yes_no(raw)
            except Exception as e:
                raw = f"ERROR: {e}"
                pred = -1

            rows.append({
                "sample_id": i,
                "ground_truth": label,      # 0=factual, 1=hallucinated
                "prediction": pred,         # 1=judge says hallucination
                "raw_output": raw,
                "question": row["question"][:200],
                "model": model,
            })

            if sleep_between:
                time.sleep(sleep_between)

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model ID as recognized by your router "
                             "(e.g. 'google/gemini-3.7-flash')")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--config", default="qa",
                        choices=["qa", "dialogue", "summarization", "general"])
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between calls (rate-limit guard)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = evaluate(
        model=args.model,
        n_samples=args.n_samples,
        out_path=args.out,
        config=args.config,
        max_tokens=args.max_tokens,
        sleep_between=args.sleep,
    )

    # Quick inline summary so user sees something immediately
    valid = df[df["prediction"] != -1]
    if len(valid):
        from src.metrics import halueval_metrics
        m = halueval_metrics(valid["ground_truth"], valid["prediction"])
        print(f"\n=== HaluEval / {args.model} ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
        unparseable = (df["prediction"] == -1).sum()
        print(f"  unparseable: {unparseable}/{len(df)}")
    print(f"\nSaved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
