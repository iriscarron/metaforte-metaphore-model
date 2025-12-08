from transformers import pipeline, XLMRobertaTokenizer
import pandas as pd
import os

def main():
    df = pd.read_csv("data/processed/polititweets_clean.csv")
    if "clean_text" in df.columns:
        df["text"] = df["clean_text"]

    # test sur echantillons 200 lignes
    df_sample = df.head(200).copy()

    texts = df_sample["text"].fillna("").tolist()

    # Chargement du tokenizer slow pour éviter le bug
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        "joeddav/xlm-roberta-large-xnli"
    )
    
    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        tokenizer=tokenizer,
        device_map="auto"  # utilisera le GPU si dispo, sinon CPU
    )

    labels = ["contient une métaphore", "ne contient pas de métaphore"]

    scores = []
    for t in texts:
        if not t.strip():
            scores.append(0.0)
            continue

        out = classifier(
            t,
            labels,
            hypothesis_template="Ce texte {}."
        )
        idx = out["labels"].index("contient une métaphore")
        scores.append(out["scores"][idx])

    df_sample["metaphor_score"] = scores

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/polititweets_with_llm_score_sample.csv"
    df_sample.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"OK - {out_path}")

if __name__ == "__main__":
    main()
