import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Utilise les scores issus de XLM-RoBERTa dans le dossier LLM
df = pd.read_csv(PROJECT_ROOT / "data/processed/llm/xlmroberta/scores.csv")

print(f"Tweets disponibles: {len(df)}")
print(f"Scores min: {df['metaphor_score'].min():.3f}, max: {df['metaphor_score'].max():.3f}, médiane: {df['metaphor_score'].median():.3f}")

df_sorted = df.sort_values("metaphor_score")

n_samples = 100
positifs = df_sorted.tail(n_samples)  # meilleur
negatifs = df_sorted.head(n_samples)  # pire score

annot = pd.concat([positifs, negatifs]).sample(frac=1, random_state=42)
annot[["text", "metaphor_score"]].to_csv(PROJECT_ROOT / "data/processed/llm/xlmroberta/polititweets_for_annotation.csv", index=False, encoding='utf-8-sig')

print(f"\n {len(positifs)} tweets avec scores élevés (top {n_samples})")
print(f"  Range: {positifs['metaphor_score'].min():.3f} - {positifs['metaphor_score'].max():.3f}")
print(f" {len(negatifs)} tweets avec scores faibles (bottom {n_samples})")
print(f"  Range: {negatifs['metaphor_score'].min():.3f} - {negatifs['metaphor_score'].max():.3f}")
print(f" Total: {len(annot)} tweets pour annotation")
print(f" Fichier: data/processed/llm/xlmroberta/polititweets_for_annotation.csv")
