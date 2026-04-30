import pandas as pd
import asyncio
import json
import re
import os
from tqdm.asyncio import tqdm
from llm_client import query_model
from dotenv import load_dotenv
from llm_client import query_model, JudgeSchema

load_dotenv()

INPUT_CSV = "comparativo_qualitativo_full.csv"
OUTPUT_CSV = "resultado_juiz_llama3.csv"
JUDGE_MODEL = "groq/llama-3.3-70b-versatile" 
CONCURRENCY_LIMIT = 2 

async def evaluate_extraction(sem, gold, pred, pico_type):
    async with sem:
        prompt = f"""
        Role: You are a Medical Evaluator.
        Task: Your task is to compare the extraction quality of PICO elements of a medical abstract (1-5).
        
        GOLD STANDARD: "{gold}"
        PREDICTION: "{pred}"
        ENTITY: {pico_type}

        Scoring: 5=Perfect/Synonym, 4=Good, 3=Ok, 2=Bad, 1=Fail. 
        """

        try:
            data = await query_model(JUDGE_MODEL, prompt, schema=JudgeSchema)
            
            return {
                "score": int(data.get("score", 1)),
                "reason": data.get("reason", "")
            }

        except Exception as e:
            print(f"\nErro na avaliação: {e}")
            return {"score": 1, "reason": f"Error: {str(e)}"}

async def main():

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['Gold_Standard', 'Predicted'])

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    for index, row in df.iterrows(): 
        task = evaluate_extraction(sem, row['Gold_Standard'], row['Predicted'], row['Type'])
        tasks.append(task)

    results = await tqdm.gather(*tasks) 

    df['Llama_Score'] = [r['score'] for r in results]
    df['Llama_Reason'] = [r['reason'] for r in results]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvo em: {OUTPUT_CSV}")

    print("\n" + "="*50)
    print("RESULTADOS FINAIS")
    print("="*50)
    summary = df.groupby(['Model', 'Type'])['Llama_Score'].mean().unstack()
    
    try:
        print(summary.to_markdown(floatfmt=".2f"))
    except ImportError:
        print(summary)

if __name__ == "__main__":
    asyncio.run(main())