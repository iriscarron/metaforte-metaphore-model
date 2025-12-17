import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import re

def extract_score_from_text(s):
    """Extrait le dernier nombre (score) d'une chaîne."""
    if not isinstance(s, str):
        return None
    numbers = re.findall(r'[\d.]+', s)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None

def evaluate():
    ann_dir = Path("data/annotations")
    ann_files = sorted(list(ann_dir.glob("*.xlsx")) + list(ann_dir.glob("*.csv")))
    
    if not ann_files:
        print(" Aucun fichier dans data/annotations/")
        return
    
    out_dir = Path("data/processed/llm/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    report_lines = []
    
    for file_path in ann_files:
        try:
            print(f"\n Chargement {file_path.name}...")
            if file_path.suffix == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            if df.shape[1] < 2:
                print(f"  Pas assez de colonnes")
                continue
            
            col1 = df.iloc[:, 0]
            col2 = df.iloc[:, 1]
            
            scores = col1.apply(extract_score_from_text)
            labels = pd.to_numeric(col2, errors="coerce")
            
            mask = ~(scores.isna() | labels.isna())
            scores = scores[mask].values
            labels = labels[mask].astype(int).values
            
            if len(scores) == 0:
                print(f"  Aucune donnée valide")
                continue
            
            print(f"  {len(scores)} samples valides")
            
            pr_prec, pr_rec, pr_thresh = precision_recall_curve(labels, scores)
            f1_curve = (2 * pr_prec * pr_rec) / (pr_prec + pr_rec + 1e-12)
            
            results[file_path.name] = {
                "scores": scores,
                "labels": labels,
                "pr_prec": pr_prec,
                "pr_rec": pr_rec,
                "pr_thresh": pr_thresh,
                "f1_curve": f1_curve,
                "n_samples": len(scores),
            }
            
            y_pred = (scores >= 0.5).astype(int)
            prec = precision_score(labels, y_pred, zero_division=0)
            rec = recall_score(labels, y_pred, zero_division=0)
            f1 = f1_score(labels, y_pred, zero_division=0)
            
            clean_name = file_path.name.replace(".xlsx", "").replace(".csv", "")
            report_lines.append(f"Model: {clean_name}")
            report_lines.append(f"  Samples: {len(scores)}")
            report_lines.append(f"  Precision @ 0.50: {prec:.4f}")
            report_lines.append(f"  Recall @ 0.50: {rec:.4f}")
            report_lines.append(f"  F1 @ 0.50: {f1:.4f}")
            report_lines.append("")
            
        except Exception as e:
            print(f"  Erreur: {e}")
    
    if not results:
        print("Impossible de charger les données.")
        return
    
    # Graphique PR
    print("\n Génération courbe PR...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in results.items():
        label = name.replace(".xlsx", "").replace(".csv", "")
        ax.plot(res["pr_rec"], res["pr_prec"], linewidth=2.5, label=label, marker="o", markersize=5)
    ax.set_xlabel("Rappel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Précision", fontsize=12, fontweight="bold")
    ax.set_title("Courbe Précision-Rappel", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    pr_path = out_dir / "pr_curves.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {pr_path}")
    
    # Graphique F1
    print(" Génération courbe F1 vs seuil...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in results.items():
        label = name.replace(".xlsx", "").replace(".csv", "")
        ax.plot(res["pr_thresh"], res["f1_curve"][:-1], linewidth=2.5, label=label, marker="s", markersize=5)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Seuil=0.50")
    ax.set_xlabel("Seuil", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1", fontsize=12, fontweight="bold")
    ax.set_title("F1 en fonction du seuil", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    f1_path = out_dir / "f1_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {f1_path}")
    
    # Rapport
    report = "\n".join(report_lines)
    report_path = out_dir / "metrics.txt"
    report_path.write_text(report)
    print(f"\nRapport sauvegardé: {report_path}")
    print(report)

if __name__ == "__main__":
    evaluate()
