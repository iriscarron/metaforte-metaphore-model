import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline


def build_classifier(model_id: str, device: int):
    return pipeline(
        "zero-shot-classification",
        model=model_id,
        device=device,
    )


def metaphor_score(classifier, text: str) -> float:
    labels = ["contient une métaphore", "ne contient pas de métaphore"]
    out = classifier(text, labels, hypothesis_template="Ce texte {}.")
    idx = out["labels"].index("contient une métaphore")
    return float(out["scores"][idx])


def occlusion_importance(classifier, text: str, top_k: int = 5):
    base = metaphor_score(classifier, text)
    words = [w for w in text.split() if w]
    if not words:
        return base, []

    impacts = []
    for i, w in enumerate(words):
        occluded = " ".join(words[:i] + words[i+1:])
        try:
            s = metaphor_score(classifier, occluded)
            # positive impact means word supports metaphor classification
            impacts.append((w, base - s))
        except Exception:
            impacts.append((w, 0.0))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return base, impacts[:top_k]


def run_for_model(model_name: str, model_id: str, input_csv: Path, out_csv: Path, n_rows: int = 50):
    device = 0 if torch.cuda.is_available() else -1
    clf = build_classifier(model_id, device)

    df = pd.read_csv(input_csv)
    if "clean_text" in df.columns:
        df["text"] = df["clean_text"]
    df = df.head(n_rows).copy()

    rows = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        if not text.strip():
            rows.append({"text": text, "base_score": 0.0, "top_words": ""})
            continue
        base, top_words = occlusion_importance(clf, text)
        rows.append({
            "text": text,
            "base_score": base,
            "top_words": ", ".join([f"{w}:{impact:+.3f}" for w, impact in top_words])
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"OK - {model_name} -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Occlusion-based word importance for metaphor detection")
    parser.add_argument("--camembert", action="store_true")
    parser.add_argument("--xlmroberta", action="store_true")
    parser.add_argument("--rows", type=int, default=50)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data/processed/polititweets_clean.csv"

    if args.camembert:
        run_for_model(
            "camembert",
            "cmarkea/distilcamembert-base-nli",
            input_path,
            project_root / "data/processed/llm/camembert/token_occlusion_sample.csv",
            n_rows=args.rows,
        )
    if args.xlmroberta:
        run_for_model(
            "xlmroberta",
            "joeddav/xlm-roberta-large-xnli",
            input_path,
            project_root / "data/processed/llm/xlmroberta/token_occlusion_sample.csv",
            n_rows=args.rows,
        )


if __name__ == "__main__":
    main()
