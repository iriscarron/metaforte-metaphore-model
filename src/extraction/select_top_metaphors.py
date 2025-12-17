import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Utilise les scores issus de XLM-RoBERTa dans le dossier LLM
df = pd.read_csv(PROJECT_ROOT / "data/processed/llm/xlmroberta/scores.csv")

print(f"Tweets disponibles: {len(df)}")
print(f"Scores min: {df['metaphor_score'].min():.3f}, max: {df['metaphor_score'].max():.3f}, médiane: {df['metaphor_score'].median():.3f}")

df_sorted = df.sort_values("metaphor_score", ascending=False)
top_200 = df_sorted.head(200)

output_path = PROJECT_ROOT / "data/processed/llm/xlmroberta/top200.csv"
top_200[["text", "metaphor_score"]].to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n{len(top_200)} tweets avec les scores les plus élevés sélectionnés")
print(f"  Range: {top_200['metaphor_score'].min():.3f} - {top_200['metaphor_score'].max():.3f}")
print(f"Fichier: data/processed/llm/xlmroberta/top200.csv")
