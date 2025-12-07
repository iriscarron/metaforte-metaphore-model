import pandas as pd
import re
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = " ".join(text.split()) #enleve retours ligne et espaces

    text = text.replace('""', '"') #enleve "" et remplace par "

    text = text.replace("\u200b", "") #caract inv

 
    text = re.sub(r"http\S+", "<URL>", text) #normalise url

    return text.strip()

def main():
    input_path = PROJECT_ROOT / "data" / "processed" / "polititweets_full.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "polititweets_clean.csv"
    
    df = pd.read_csv(input_path)

    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv(output_path, index=False)

    print(f"Tweets nettoyés ok dans {output_path}")

if __name__ == "__main__":
    main()
