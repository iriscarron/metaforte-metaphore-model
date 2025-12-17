from transformers import pipeline
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    input_path = PROJECT_ROOT / "data/processed/polititweets_clean.csv"
    df = pd.read_csv(input_path)
    
    if "clean_text" in df.columns:
        df["text"] = df["clean_text"]
    
    df_sample = df.head(200).copy()
    
    print(f"analyse de {len(df_sample)} tweets avec camembert...")
    
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "zero-shot-classification",
        model="cmarkea/distilcamembert-base-nli",
        device=device
    )
    
    labels = ["contient une métaphore", "ne contient pas de métaphore"]
    
    texts = df_sample["text"].fillna("").tolist()
    
    scores = []
    for idx, t in enumerate(texts):
        if not t.strip():
            scores.append(0.0)
            continue
        
        out = classifier(
            t,
            labels,
            hypothesis_template="Ce texte {}."
        )
        idx_label = out["labels"].index("contient une métaphore")
        scores.append(out["scores"][idx_label])
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(texts)} tweets analysés")
    
    df_sample["metaphor_score"] = scores

    # Dossiers par LLM
    llm_dir = PROJECT_ROOT / "data/processed/llm/camembert"
    llm_dir.mkdir(parents=True, exist_ok=True)

    # CSV principal avec scores CamemBERT (nom explicite par LLM)
    output_path = llm_dir / "scores.csv"
    df_sample.to_csv(output_path, index=False, encoding='utf-8-sig')

    # CSV finals: top 100 (plus fortes) et bottom 100 (plus faibles)
    df_sorted_desc = df_sample.sort_values("metaphor_score", ascending=False)
    top_100 = df_sorted_desc.head(100)[["text", "metaphor_score"]]
    bottom_100 = df_sorted_desc.tail(100).sort_values("metaphor_score", ascending=True)[["text", "metaphor_score"]]

    top_out = llm_dir / "top100.csv"
    bottom_out = llm_dir / "bottom100.csv"
    top_100.to_csv(top_out, index=False, encoding='utf-8-sig')
    bottom_100.to_csv(bottom_out, index=False, encoding='utf-8-sig')

    print(f"\nrésultats sauvegardés dans {output_path}")
    print(f"top sauvegardés dans {top_out}")
    print(f"bottom sauvegardés dans {bottom_out}")
    print(f"score moyen: {df_sample['metaphor_score'].mean():.3f}")

if __name__ == "__main__":
    main()
