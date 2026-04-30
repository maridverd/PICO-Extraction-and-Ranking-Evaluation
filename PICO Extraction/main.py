import os
import asyncio
import json
import csv
import glob
from tqdm import tqdm
from dotenv import load_dotenv
from llm_client import query_model, PicoSchema
from metrics import score_field
from pico_classifier import PicoClassifier

PICO_PROMPT = """
You are a biomedical extraction system. 
Your task is to identify and copy EXACT segments from the abstract that correspond to the PICO elements.

Rules:
1. EXTRACT VERBATIM: Do not paraphrase, summarize, or modify the text. Copy the exact words from the abstract.
2. PREFER LONG SPANS: If an intervention includes dosage, route, or frequency, include the whole phrase.

Abstract:
{{ABSTRACT}}
"""

MODELS = [
    {
        "name": "llama3_8b_v3",
        "model": "groq/llama-3.1-8b-instant" 
    },
    {
        "name": "llama3_70b",
        "model": "groq/llama-3.3-70b-versatile"
    }
]


load_dotenv()

BASE_DIR = "/EBM-NLP/ebm_nlp_2_00"

DOC_DIR = os.path.join(BASE_DIR, "documents")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations", "aggregated", "starting_spans")
GOLD_DIRS = {
    "participants": os.path.join(ANNOTATIONS_DIR, "participants", "test", "gold"),
    "interventions": os.path.join(ANNOTATIONS_DIR, "interventions", "test", "gold"),
    "outcomes": os.path.join(ANNOTATIONS_DIR, "outcomes", "test", "gold")
}

OUTPUT_METRICS = "comparativo_metrics_full.csv"
OUTPUT_QUALITATIVE = "comparativo_qualitativo_full.csv"

def read_tokens(pmid):
    clean_id = pmid.replace(".AGGREGATED", "") 
    token_path = os.path.join(DOC_DIR, f"{clean_id}.tokens") 
    if not os.path.exists(token_path):
        token_path = os.path.join(DOC_DIR, f"{clean_id}.text")

    if os.path.exists(token_path):
        with open(token_path, 'r', encoding='utf-8', errors='ignore') as f:
            tokens = [line.strip() for line in f.readlines()]
            full_text = " ".join(tokens) 
            return tokens, full_text
    return [], ""

def get_gold_extracted_text(pmid, pico_type, tokens):
    path_v1 = os.path.join(GOLD_DIRS[pico_type], f"{pmid}.ann")
    path_v2 = os.path.join(GOLD_DIRS[pico_type], f"{pmid}.AGGREGATED.ann")
    ann_path = path_v1 if os.path.exists(path_v1) else path_v2
    
    extracted_spans = []
    if os.path.exists(ann_path) and tokens: 
        with open(ann_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()] 
            
        limit = min(len(labels), len(tokens))
        current_span = []
        for i in range(limit):
            if labels[i] == '1': 
                current_span.append(tokens[i])
            else: 
                if current_span: 
                    extracted_spans.append(" ".join(current_span))
                    current_span = []
        if current_span: extracted_spans.append(" ".join(current_span)) 
                
    return extracted_spans

async def main():
    with open(OUTPUT_METRICS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PICO_Type", "Precision", "Recall", "F1", "Docs_Processed"])

    q_file = open(OUTPUT_QUALITATIVE, 'w', newline='', encoding='utf-8')
    q_writer = csv.writer(q_file)
    q_writer.writerow(["PMID", "Model", "Type", "Gold_Standard", "Predicted", "TP", "FN", "FP"])


    bio_electra = PicoClassifier()
    llm_cfg = MODELS[1]["model"]


    raw_files = glob.glob(os.path.join(GOLD_DIRS['participants'], "*.ann"))
    test_ids = list(set([os.path.basename(f).replace(".ann", "").replace(".AGGREGATED", "") for f in raw_files]))
    
    metrics_acc = {
        "LLM": {"participants": [], "interventions": [], "outcomes": []},
        "NER": {"participants": [], "interventions": [], "outcomes": []}
    }

    for pmid in tqdm(test_ids):
        tokens, abstract = read_tokens(pmid)
        if not abstract or len(abstract) < 20: continue

        try:
            prompt_completo = PICO_PROMPT.replace("{{ABSTRACT}}", abstract)
            llm_pico = await query_model(llm_cfg, prompt_completo, schema=PicoSchema)
            await asyncio.sleep(3) 
            
        except Exception as e: 
            print(f"\nErro na extração: {e}")
            llm_pico = {"POPULATION": [], "INTERVENTION": [], "OUTCOME": []}

        try:
            ner_raw = bio_electra._extract_pico_from_text(abstract)
        except: ner_raw = {}

        types_to_eval = ["participants", "interventions", "outcomes"]
        
        for p_type in types_to_eval:
            gold_list = get_gold_extracted_text(pmid, p_type, tokens)
            
            if not gold_list: continue

            llm_preds = []
            ner_preds = []

            if p_type == "participants":
                llm_preds = llm_pico.get("POPULATION", [])
                ner_preds = ner_raw.get("I-Population", []) + ner_raw.get("I-Participants", [])
            
            elif p_type == "interventions":
                llm_preds = llm_pico.get("INTERVENTION", []) + llm_pico.get("COMPARISON", [])
                ner_preds = ner_raw.get("I-Intervention", []) + ner_raw.get("I-Comparator", []) 
            
            elif p_type == "outcomes":
                llm_preds = llm_pico.get("OUTCOME", [])
                ner_preds = ner_raw.get("I-Outcome", [])

            res_llm = score_field(llm_preds, gold_list)
            metrics_acc["LLM"][p_type].append(res_llm)
            
            res_ner = score_field(ner_preds, gold_list)
            metrics_acc["NER"][p_type].append(res_ner)

            q_writer.writerow([pmid, "LLM", p_type, " | ".join(gold_list), " | ".join(llm_preds), res_llm["tp"], res_llm["fn"], res_llm["fp"]])
            q_writer.writerow([pmid, "NER", p_type, " | ".join(gold_list), " | ".join(ner_preds), res_ner["tp"], res_ner["fn"], res_ner["fp"]])

    q_file.close()

    print("\nRESULTADOS FINAIS:")
    for model_name in ["LLM", "NER"]:
        for p_type in ["participants", "interventions", "outcomes"]:
            results = metrics_acc[model_name][p_type]
            if not results: continue
            
            avg_prec = sum(r['precision'] for r in results) / len(results)
            avg_rec = sum(r['recall'] for r in results) / len(results)
            avg_f1 = sum(r['f1'] for r in results) / len(results)
            
            print(f"    {model_name} [{p_type:15}]: F1={avg_f1:.4f} (P={avg_prec:.4f}, R={avg_rec:.4f})")
            
            with open(OUTPUT_METRICS, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, p_type, f"{avg_prec:.4f}", f"{avg_rec:.4f}", f"{avg_f1:.4f}", len(results)])

if __name__ == "__main__":
    asyncio.run(main())