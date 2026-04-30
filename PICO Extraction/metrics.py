import re

def normalize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())

def normalize_string(text: str):
    text = str(text).lower()
    return re.sub(r"[^a-z0-9\s]", " ", text).strip()

def semantic_match(pred: str, gold: str, threshold=0.5) -> bool:
    """
    Critérios de Acerto:
    1. Jaccard (Sobreposição) >= threshold
    2. Containment (Um está contido no outro)
    """
    p_set = normalize(pred)
    g_set = normalize(gold)
    
    p_str = normalize_string(pred)
    g_str = normalize_string(gold)

    if not p_set or not g_set:
        return False

    intersection = len(p_set & g_set) 
    union = len(p_set | g_set) 
    jaccard = intersection / union if union > 0 else 0 

    is_contained = (p_str in g_str) or (g_str in p_str) and (len(p_str) > 2 and len(g_str) > 2) 

    return (jaccard >= threshold) or is_contained 

def score_field(preds, golds):
    preds = preds if isinstance(preds, list) else []
    golds = golds if isinstance(golds, list) else []

    preds_unique = list(set(preds))
    golds_unique = list(set(golds))

    if not preds_unique and not golds_unique:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = 0
    matched_gold_indices = set()

    for p in preds_unique: 
        match_found = False
        for i, g in enumerate(golds_unique):
            if i not in matched_gold_indices: 
                if semantic_match(p, g): 
                    tp += 1
                    matched_gold_indices.add(i) 
                    match_found = True
                    break

    fp = len(preds_unique) - tp
    fn = len(golds_unique) - len(matched_gold_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn
    }