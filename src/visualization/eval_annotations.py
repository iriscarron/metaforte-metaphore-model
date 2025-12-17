import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(lambda s: " ".join(s.split()).lower())


def _load_annotations(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Le fichier d'annotation doit contenir au moins deux colonnes: texte et label (0/1).")

    text_col = df.columns[0]
    label_col = df.columns[1]
    df = df.rename(columns={text_col: "text", label_col: "label"})

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text_norm"] = _normalize_text(df["text"])
    return df[["text", "text_norm", "label"]]


def _load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "metaphor_score" not in df.columns:
        raise ValueError(f"metaphor_score manquant dans {path}")
    if "text" not in df.columns:
        # si la colonne texte a été renommée différemment
        text_col = df.columns[0]
        df = df.rename(columns={text_col: "text"})
    df["text_norm"] = _normalize_text(df["text"])
    return df[["text", "text_norm", "metaphor_score"]]


def evaluate_model(name: str, scores: pd.DataFrame, ann: pd.DataFrame, threshold: float):
    merged = scores.merge(ann, on="text_norm", how="inner", suffixes=("_pred", "_gold"))
    if merged.empty:
        raise ValueError(f"Aucune correspondance texte trouvée pour {name}. Vérifiez que les textes sont identiques (ou quasi-identiques) entre annotations et scores.")

    y_true = merged["label"].values
    y_score = merged["metaphor_score"].values
    y_pred = (y_score >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_score)
    f1_curve = (2 * pr_prec * pr_rec) / (pr_prec + pr_rec + 1e-12)

    return {
        "name": name,
        "n_samples": len(merged),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_prec": pr_prec,
        "pr_rec": pr_rec,
        "pr_thresh": pr_thresh,
        "f1_curve": f1_curve,
    }


def plot_curves(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for r in results:
        plt.plot(r["pr_rec"], r["pr_prec"], label=f"{r['name']}")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbes Précision-Rappel")
    plt.legend()
    pr_path = out_dir / "pr_curves.png"
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # F1 vs threshold
    plt.figure(figsize=(8, 6))
    for r in results:
        # pr_thresh a une longueur de n-1 par rapport aux points pr_prec/pr_rec; on alignera avec f1_curve[:-1]
        plt.plot(r["pr_thresh"], r["f1_curve"][:-1], label=f"{r['name']}")
    plt.xlabel("Seuil")
    plt.ylabel("F1")
    plt.title("F1 en fonction du seuil")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    f1_path = out_dir / "f1_vs_threshold.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()

    return pr_path, f1_path


def main():
    parser = argparse.ArgumentParser(description="Évalue les modèles avec annotations 0/1 et génère courbes PR/F1.")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotated.csv"), help="CSV annoté (colonne texte + colonne label 0/1)")
    parser.add_argument("--camembert-scores", type=Path, default=Path("data/processed/llm/camembert/scores.csv"), help="CSV scores CamemBERT")
    parser.add_argument("--xlmroberta-scores", type=Path, default=Path("data/processed/llm/xlmroberta/scores.csv"), help="CSV scores XLM-RoBERTa")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil pour calculer précision/rappel/F1")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/llm/reports"), help="Dossier de sortie pour rapports et figures")

    args = parser.parse_args()

    ann = _load_annotations(args.annotations)

    results = []
    for name, score_path in [("camembert", args.camembert_scores), ("xlmroberta", args.xlmroberta_scores)]:
        scores = _load_scores(score_path)
        res = evaluate_model(name, scores, ann, threshold=args.threshold)
        results.append(res)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for r in results:
        lines.append(f"Model: {r['name']}")
        lines.append(f"  Samples: {r['n_samples']}")
        lines.append(f"  Precision@{args.threshold}: {r['precision']:.3f}")
        lines.append(f"  Recall@{args.threshold}: {r['recall']:.3f}")
        lines.append(f"  F1@{args.threshold}: {r['f1']:.3f}")
        lines.append("")
    report_path = args.output_dir / "metrics.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    pr_path, f1_path = plot_curves(results, args.output_dir)

    print("Rapport écrit :", report_path)
    print("Courbe PR :", pr_path)
    print("Courbe F1 :", f1_path)


if __name__ == "__main__":
    main()
