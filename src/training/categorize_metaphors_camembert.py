from transformers import pipeline, CamembertTokenizer
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    input_path = PROJECT_ROOT / "data/processed/polititweets_with_llm_score_sample.csv"
    df = pd.read_csv(input_path)
    
    df_metaphors = df[df['metaphor_score'] > 0.7].copy()
    
    print(f"analyse de {len(df_metaphors)} tweets avec métaphores...")
    
    tokenizer = CamembertTokenizer.from_pretrained(
        "BaptisteDoyen/camembert-base-xnli"
    )
    
    classifier = pipeline(
        "zero-shot-classification",
        model="BaptisteDoyen/camembert-base-xnli",
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    categories = [
        "eau",
        "feu",
        "guerre et combat",
        "construction",
        "voyage",
        "nature",
        "corps",
        "mouvement",
        "lumière",
        "technologie"
    ]
    
    texts = df_metaphors["text"].fillna("").tolist()
    
    results = []
    for idx, text in enumerate(texts):
        if not text.strip():
            results.append({
                'category': 'non classifiable',
                'score': 0.0
            })
            continue
        
        out = classifier(
            text,
            categories,
            hypothesis_template="Ce texte contient une métaphore liée à {}.",
            multi_label=False
        )
        
        results.append({
            'category': out["labels"][0],
            'score': out["scores"][0]
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(texts)} tweets analysés")
    
    df_metaphors['metaphor_category'] = [r['category'] for r in results]
    df_metaphors['category_score'] = [r['score'] for r in results]
    
    output_path = PROJECT_ROOT / "data/processed/polititweets_camembert_categories.csv"
    df_metaphors.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nrésultats sauvegardés dans {output_path}")
    print(f"\nrépartition des catégories:")
    print(df_metaphors['metaphor_category'].value_counts())

if __name__ == "__main__":
    main()
