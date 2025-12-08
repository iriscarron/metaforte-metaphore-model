from transformers import pipeline, CamembertTokenizer
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
    
    tokenizer = CamembertTokenizer.from_pretrained(
        "BaptisteDoyen/camembert-base-xnli"
    )
    
    classifier = pipeline(
        "zero-shot-classification",
        model="BaptisteDoyen/camembert-base-xnli",
        tokenizer=tokenizer,
        device_map="auto"
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
    
    output_path = PROJECT_ROOT / "data/processed/polititweets_camembert_score.csv"
    df_sample.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nrésultats sauvegardés dans {output_path}")
    print(f"score moyen: {df_sample['metaphor_score'].mean():.3f}")

if __name__ == "__main__":
    main()
