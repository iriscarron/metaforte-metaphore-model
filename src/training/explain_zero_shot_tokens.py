import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

NLI_ENTAILMENT_INDEX = 2  # for most RoBERTa/XLM-RoBERTa-based NLI heads


def tokenize_pair(tokenizer, text: str, hypothesis: str):
    encoded = tokenizer(text, hypothesis, return_tensors="pt")
    return encoded


def get_entailment_logit(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits  # [batch, 3]
    return logits[:, NLI_ENTAILMENT_INDEX]


def compute_attributions(model, tokenizer, text: str, hypothesis: str, steps: int = 25):
    model.eval()
    inputs = tokenize_pair(tokenizer, text, hypothesis)
    for k in inputs:
        inputs[k] = inputs[k]
    inputs = {k: v for k, v in inputs.items()}

    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits[:, NLI_ENTAILMENT_INDEX]

    ig = IntegratedGradients(forward_func)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Baseline = zeros input_ids; Captum uses embeddings internally for IDs
    attributions, _ = ig.attribute(inputs=(input_ids, attention_mask),
                                   baselines=(torch.zeros_like(input_ids), attention_mask),
                                   n_steps=steps, return_convergence_delta=True)
    # Sum across embedding dims if present; for IDs, each token returns a scalar attribution
    # Here, Captum returns shape [batch, seq_len]; keep token-level attribution
    token_attr = attributions.squeeze(0).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    # Identify boundary between premise and hypothesis using '</s>' tokens
    sep_indices = [i for i, t in enumerate(tokens) if t == '</s>']
    # RoBERTa pair format: <s> premise </s> </s> hypothesis </s>
    if len(sep_indices) >= 2:
        premise_start = 1
        premise_end = sep_indices[0]
    else:
        premise_start = 1
        premise_end = len(tokens)

    # Aggregate subword attributions to words (SentencePiece uses '▁' for word boundaries)
    word_attrs = []
    current_word = ''
    current_attr = 0.0
    for i in range(premise_start, premise_end):
        tok = tokens[i]
        attr = float(token_attr[i])
        # Normalize CamemBERT/XLM-RoBERTa token forming
        if tok.startswith('▁'):
            # push previous word
            if current_word:
                word_attrs.append((current_word, current_attr))
            current_word = tok[1:]
            current_attr = attr
        else:
            # continuation of word
            current_word += tok.replace('Ġ', '')
            current_attr += attr
    if current_word:
        word_attrs.append((current_word, current_attr))

    # Sort by absolute importance
    word_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_attrs[:5]  # top 5 words


def explain_file(model_id: str, input_csv: Path, output_csv: Path, label_positive_fr: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    df = pd.read_csv(input_csv)
    if "clean_text" in df.columns:
        df["text"] = df["clean_text"]
    df = df.head(50).copy()  # limit for speed

    hypothesis = f"Ce texte {label_positive_fr}."

    rows = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        if not text.strip():
            rows.append({"text": text, "top_tokens": ""})
            continue
        try:
            top_tokens = compute_attributions(model, tokenizer, text, hypothesis)
            rows.append({
                "text": text,
                "top_tokens": ", ".join([f"{w}:{a:.3f}" for w, a in top_tokens])
            })
        except Exception as e:
            rows.append({"text": text, "top_tokens": f"error: {e}"})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"OK - {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camembert", action="store_true")
    parser.add_argument("--xlmroberta", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data/processed/polititweets_clean.csv"

    if args.camembert:
        explain_file(
            model_id="cmarkea/distilcamembert-base-nli",
            input_csv=input_path,
            output_csv=project_root / "data/processed/llm/camembert/token_explanations_sample.csv",
            label_positive_fr="contient une métaphore"
        )
    if args.xlmroberta:
        explain_file(
            model_id="joeddav/xlm-roberta-large-xnli",
            input_csv=input_path,
            output_csv=project_root / "data/processed/llm/xlmroberta/token_explanations_sample.csv",
            label_positive_fr="contient une métaphore"
        )


if __name__ == "__main__":
    main()
