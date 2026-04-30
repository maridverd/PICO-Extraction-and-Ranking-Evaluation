from llm_client import query_model
from llm_client import InclusionSchema 

async def classify_inclusion_async(target_pico, abstract, model_name):
    prompt = f"""
    TARGET PICO:
    - Population: {target_pico.get('PART', 'Any')}
    - Intervention: {target_pico.get('INT', 'Any')}
    - Outcome: {target_pico.get('OUT', 'Any')}

    CANDIDATE ABSTRACT:
    {abstract}

    TASK:
    Does the candidate abstract match the Target PICO topics? 
    It does NOT need to be a perfect match. If it is related, mark as INCLUDE.
    """

    try:
        data = await query_model(model_name, prompt, schema=InclusionSchema)
        return {"decision": data.get("decision"), "reason": data.get("reason")}

    except Exception as e:
        return {"decision": "INCLUDE", "reason": "Error fallback"}