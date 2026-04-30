import os
import asyncio
from litellm import acompletion
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal
import json

load_dotenv()

class PicoSchema(BaseModel):
    POPULATION: List[str] = Field(default=[], description="List of participants, patients, or population characteristics.")
    INTERVENTION: List[str] = Field(default=[], description="List of interventions, treatments, or comparators.")
    OUTCOME: List[str] = Field(default=[], description="List of measured outcomes or results.")

class InclusionSchema(BaseModel):
    decision: Literal["INCLUDE", "EXCLUDE"] = Field(description="Decision to include or exclude the candidate abstract.")
    reason: str = Field(description="Short explanation for the decision.")

async def query_model(model_name, prompt, schema=None, retries=5):
    messages = [
        {"role": "user", "content": prompt}
    ]

    for attempt in range(retries):
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
                "timeout": 60
            }
            
            if schema: 
                kwargs["response_format"] = schema

            response = await acompletion(**kwargs)
            content = response.choices[0].message.content
            
            if schema:
                return json.loads(content)
            
            return content

        except Exception as e:
            error_msg = str(e).lower()
            if attempt == retries - 1:
                return {} if schema else ""
            
            if "rate limit" in error_msg or "429" in error_msg:
                await asyncio.sleep(20 + (attempt * 5))
            else:
                await asyncio.sleep(2)
            
    return {} if schema else ""