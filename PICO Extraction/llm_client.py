import os
import json
from dotenv import load_dotenv
from litellm import acompletion
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


class PicoSchema(BaseModel):
    POPULATION: List[str] = Field(default=[], description="Exact segments of participants, patients, or population characteristics. Extract verbatim.")
    INTERVENTION: List[str] = Field(default=[], description="Exact segments of interventions, treatments, or comparators. Extract verbatim.")
    OUTCOME: List[str] = Field(default=[], description="Exact segments of measured outcomes or results. Extract verbatim.")

class JudgeSchema(BaseModel):
    score: int = Field(description="Score from 1 to 5 rating the extraction quality.")
    reason: str = Field(description="Short explanation for why this score was given.")

@retry(
    retry=retry_if_exception_type((Exception)), 
    wait=wait_exponential(multiplier=1, min=2, max=60), 
    stop=stop_after_attempt(5) 
)
async def query_model(model_name: str, prompt: str, schema=None): 
    try:
        if "gemini" in model_name and "GEMINI_API_KEY" not in os.environ:
             if os.getenv("GOOGLE_API_KEY"):
                 os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        if schema:
            kwargs["response_format"] = schema

        response = await acompletion(**kwargs)
        
        content = response["choices"][0]["message"]["content"]
        
        if schema:
            return json.loads(content)
            
        return content

    except Exception as e:
        print(f"Erro na chamada de {model_name}: {e}")
        raise e