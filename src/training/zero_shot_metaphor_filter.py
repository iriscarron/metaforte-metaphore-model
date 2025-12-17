from transformers import pipeline, XLMRobertaTokenizer
import torch
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
    
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        tokenizer=tokenizer,
        device=device  # 0=GPU, -1=CPU
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

    # Dossiers par LLM
    llm_dir = "data/processed/llm/xlmroberta"
    os.makedirs(llm_dir, exist_ok=True)

    # CSV principal avec scores XLM-RoBERTa (nom explicite par LLM)
    out_path = f"{llm_dir}/scores.csv"
    df_sample.to_csv(out_path, index=False, encoding='utf-8-sig')

    # CSV finals: top 100 et bottom 100
    df_sorted_desc = df_sample.sort_values("metaphor_score", ascending=False)
    top_100 = df_sorted_desc.head(100)[["text", "metaphor_score"]]
    bottom_100 = df_sorted_desc.tail(100).sort_values("metaphor_score", ascending=True)[["text", "metaphor_score"]]

    top_out = f"{llm_dir}/top100.csv"
    bottom_out = f"{llm_dir}/bottom100.csv"
    top_100.to_csv(top_out, index=False, encoding='utf-8-sig')
    bottom_100.to_csv(bottom_out, index=False, encoding='utf-8-sig')

    print(f"OK - {out_path}")
    print(f"OK - {top_out}")
    print(f"OK - {bottom_out}")

if __name__ == "__main__":
    main()
