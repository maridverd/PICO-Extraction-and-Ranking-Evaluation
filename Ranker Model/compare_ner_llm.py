import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pico_classifier import PicoClassifier
from rank_papers import Ranker


DATASET_FILE = "ranking_gold_dataset.json"

def adapt_ner_to_ranker(raw_pico):
    mapped_pico = {
        "I-Participants": [],
        "I-Intervention": [],
        "I-Outcome": []
    }

    if not raw_pico:
        return mapped_pico

    for key, values in raw_pico.items():
        key_upper = key.upper()
        if "POPULATION" in key_upper or "PARTICIPANT" in key_upper:
            mapped_pico["I-Participants"].extend(values)
        elif "INTERVENTION" in key_upper or "COMPARATOR" in key_upper:
            mapped_pico["I-Intervention"].extend(values)
        elif "OUTCOME" in key_upper:
            mapped_pico["I-Outcome"].extend(values)
            
    return mapped_pico

def evaluate_method(dataset, method_name, ner_model, ranker_model):
    precisions_at_1 = []
    precisions_at_5 = []
    recalls_at_10 = [] 
    
    for entry in tqdm(dataset):
        query_pico = entry["query_pico"]
        candidates = entry["candidates"]
        
        total_relevant_in_query = sum(1 for c in candidates if c["label"] == 1)
        
        if total_relevant_in_query == 0:
            continue

        papers_dict = {}
        ground_truth = {}
        
        for cand in candidates:
            cand_id = cand["Id"]
            ground_truth[cand_id] = cand["label"]
            
            if method_name == "LLM (Llama 3)":
                extracted_pico = cand.get("picos", {})
            else:
                abstract_text = cand.get("AbstractText", "")
                if not abstract_text:
                    raw_ner = {}
                else:
                    raw_ner = ner_model._extract_pico_from_text(abstract_text)
                
                extracted_pico = adapt_ner_to_ranker(raw_ner)

            papers_dict[cand_id] = {
                'ArticleTitle': f"PMID {cand_id}",
                'doi': '',
                'picos': extracted_pico
            }

        if not papers_dict:
            continue
            
        ranked_df = ranker_model.rank_papers_by_similarity(query_pico, papers_dict)
        
        hits_at_5 = 0
        hits_at_10 = 0
        top1_hit = 0
        
        ranked_ids = ranked_df['Id'].tolist()
        
        for i, pid in enumerate(ranked_ids[:5]): 
            is_relevant = ground_truth.get(pid, 0)
            if is_relevant == 1:
                hits_at_5 += 1
                if i == 0: 
                    top1_hit = 1
        
        for pid in ranked_ids[:10]:
            if ground_truth.get(pid, 0) == 1:
                hits_at_10 += 1
        
        precisions_at_1.append(top1_hit)
        
        denom_p5 = min(5, len(candidates))
        p5 = hits_at_5 / denom_p5 if denom_p5 > 0 else 0
        precisions_at_5.append(p5)
        
        recall = hits_at_10 / total_relevant_in_query
        recalls_at_10.append(recall)

    return np.mean(precisions_at_1), np.mean(precisions_at_5), np.mean(recalls_at_10)

def main():
    with open(DATASET_FILE, "r") as f:
        data = json.load(f)

    ner = PicoClassifier()
    ranker = Ranker()

    llm_p1, llm_p5, llm_r10 = evaluate_method(data, "LLM (Llama 3)", ner, ranker)
    ner_p1, ner_p5, ner_r10 = evaluate_method(data, "NER (BioELECTRA)", ner, ranker)

    print("\n" + "="*60)
    print("LLM vs NER")
    print("="*60)
    print(f"{'Método':<20} | {'P@1':<8} | {'P@5':<8} | {'Recall@10':<8}")
    print("-" * 60)
    print(f"{'LLM (Llama 3)':<20} | {llm_p1:.4f}   | {llm_p5:.4f}   | {llm_r10:.4f}")
    print(f"{'NER (BioELECTRA)':<20} | {ner_p1:.4f}   | {ner_p5:.4f}   | {ner_r10:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()