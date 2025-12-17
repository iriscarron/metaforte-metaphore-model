import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Évaluation Métaphores", layout="wide")
st.title("Évaluation Précision/Rappel/F1")

ann_dir = Path("data/annotations")
ann_files = sorted(list(ann_dir.glob("*.xlsx")) + list(ann_dir.glob("*.csv")))

if not ann_files:
    st.error("Aucun fichier dans data/annotations/")
    st.stop()

st.sidebar.markdown("### Fichiers d'annotation disponibles")
for f in ann_files:
    st.sidebar.text(f.name)

results = {}
for file_path in ann_files:
    try:
        if file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        score_col = df.columns[0]
        label_col = df.columns[1]
        
        scores = pd.to_numeric(df[score_col], errors="coerce")
        labels = pd.to_numeric(df[label_col], errors="coerce")
        
        mask = ~(scores.isna() | labels.isna())
        scores = scores[mask].values
        labels = labels[mask].astype(int).values
        
        if len(scores) == 0:
            st.warning(f" {file_path.name}: aucune donnée valide")
            continue
        
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
        
    except Exception as e:
        st.error(f"❌ {file_path.name}: {e}")

if not results:
    st.error("Impossible de charger les fichiers d'annotation.")
    st.stop()

threshold = st.slider("Seuil de classification", 0.0, 1.0, 0.5, step=0.01)

st.subheader(f"Métriques @ seuil = {threshold:.2f}")
cols = st.columns(len(results))
for idx, (name, res) in enumerate(results.items()):
    y_pred = (res["scores"] >= threshold).astype(int)
    prec = precision_score(res["labels"], y_pred, zero_division=0)
    rec = recall_score(res["labels"], y_pred, zero_division=0)
    f1 = f1_score(res["labels"], y_pred, zero_division=0)
    
    with cols[idx]:
        st.metric(name.replace(".xlsx", "").replace(".csv", ""), f"F1", f"{f1:.3f}")
        st.metric("Précision", f"{prec:.3f}")
        st.metric("Rappel", f"{rec:.3f}")
        st.metric("Samples", res["n_samples"])

# Courbes PR
st.subheader(" Courbes Précision-Rappel")
fig, ax = plt.subplots(figsize=(10, 6))
for name, res in results.items():
    label = name.replace(".xlsx", "").replace(".csv", "")
    ax.plot(res["pr_rec"], res["pr_prec"], linewidth=2.5, label=label, marker="o", markersize=4)
ax.set_xlabel("Rappel", fontsize=12)
ax.set_ylabel("Précision", fontsize=12)
ax.set_title("Courbe Précision-Rappel", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
st.pyplot(fig, use_container_width=True)

# F1 vs seuil
st.subheader("📈 F1 en fonction du seuil")
fig, ax = plt.subplots(figsize=(10, 6))
for name, res in results.items():
    label = name.replace(".xlsx", "").replace(".csv", "")
    ax.plot(res["pr_thresh"], res["f1_curve"][:-1], linewidth=2.5, label=label, marker="s", markersize=4)
ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Seuil={threshold:.2f}")
ax.set_xlabel("Seuil", fontsize=12)
ax.set_ylabel("F1", fontsize=12)
ax.set_title("F1 vs Seuil", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
st.pyplot(fig, use_container_width=True)

# Rapport
st.subheader(" Rapport complet")
report_lines = []
for name, res in results.items():
    y_pred = (res["scores"] >= threshold).astype(int)
    prec = precision_score(res["labels"], y_pred, zero_division=0)
    rec = recall_score(res["labels"], y_pred, zero_division=0)
    f1 = f1_score(res["labels"], y_pred, zero_division=0)
    
    clean_name = name.replace(".xlsx", "").replace(".csv", "")
    report_lines.append(f"Model: {clean_name}")
    report_lines.append(f"  Samples: {res['n_samples']}")
    report_lines.append(f"  Precision @ {threshold:.2f}: {prec:.4f}")
    report_lines.append(f"  Recall @ {threshold:.2f}: {rec:.4f}")
    report_lines.append(f"  F1 @ {threshold:.2f}: {f1:.4f}")
    report_lines.append("")

report = "\n".join(report_lines)
st.text_area("Résultats", value=report, height=250, disabled=True)

if st.button(" Sauvegarder rapport"):
    out_path = Path("data/processed/llm/reports/metrics.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    st.success(f" Rapport sauvegardé : {out_path}")
