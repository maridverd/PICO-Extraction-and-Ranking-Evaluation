import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

INPUT_CSV = "comparativo_qualitativo_full.csv"
#MODEL_NAME = "kamalkraj/BioSimCSE-BioLinkBERT-BASE" # Utilizando o mesmo modelo do ranker
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" # Fine-tuned para trazer sinónimos mais pertos no espaço vetorial

def load_data():
    try:
        df = pd.read_csv(INPUT_CSV)
        df = df.dropna(subset=['Gold_Standard', 'Predicted'])
        return df
    except FileNotFoundError:
        exit()
     
def calculate_bertscore(model, gold_str, pred_str):

    golds = [g.strip() for g in str(gold_str).split("|") if len(g.strip()) > 2]
    preds = [p.strip() for p in str(pred_str).split("|") if len(p.strip()) > 2]

    if not golds and not preds: return 1.0, 1.0, 1.0 
    if not golds or not preds: return 0.0, 0.0, 0.0  

    emb_golds = model.encode(golds, convert_to_tensor=True, show_progress_bar=False)
    emb_preds = model.encode(preds, convert_to_tensor=True, show_progress_bar=False)

    cosine_scores = util.cos_sim(emb_preds, emb_golds)

    precision_scores, _ = cosine_scores.max(dim=1) 
    semantic_precision = float(precision_scores.mean())

    recall_scores, _ = cosine_scores.max(dim=0)
    semantic_recall = float(recall_scores.mean())

    if (semantic_precision + semantic_recall) > 0:
        semantic_f1 = 2 * (semantic_precision * semantic_recall) / (semantic_precision + semantic_recall)
    else:
        semantic_f1 = 0.0

    return semantic_precision, semantic_recall, semantic_f1

def main():
    model = SentenceTransformer(MODEL_NAME)
    
    df = load_data()
    print(f"Analisando {len(df)} registros do CSV...")

    results = {
        "LLM": {"participants": [], "interventions": [], "outcomes": []},
        "NER": {"participants": [], "interventions": [], "outcomes": []}
    }

    for index, row in tqdm(df.iterrows(), total=len(df)):
        model_name = row['Model'] 
        pico_type = row['Type']  
        
        gold = row['Gold_Standard']
        pred = row['Predicted']
        
        p, r, f1 = calculate_bertscore(model, gold, pred)
        
        results[model_name][pico_type].append({
            "P": p, "R": r, "F1": f1
        })

    print("\n" + "="*60)
    print(f"RESULTADOS")
    print("="*60)
    print(f"{'Modelo':<5} | {'Tipo':<15} | {'Sem. Precision':<15} | {'Sem. Recall':<15} | {'Sem. F1':<10}")
    print("-" * 65)

    for model_name in ["LLM", "NER"]:
        for p_type in ["participants", "interventions", "outcomes"]:
            data = results[model_name][p_type]
            if not data:
                continue

            p_vals = [x['P'] for x in data]
            r_vals = [x['R'] for x in data]
            f1_vals = [x['F1'] for x in data]

            avg_p = np.mean(p_vals)
            avg_r = np.mean(r_vals)
            avg_f1 = np.mean(f1_vals)

            std_p = np.std(p_vals)
            std_r = np.std(r_vals)
            std_f1 = np.std(f1_vals)

            print(
                f"{model_name:<5} | {p_type:<15} | "
                f"{avg_p:.4f} ± {std_p:.4f} | "
                f"{avg_r:.4f} ± {std_r:.4f} | "
                f"{avg_f1:.4f} ± {std_f1:.4f}"
            )

    print("="*60)

if __name__ == "__main__":
    main()