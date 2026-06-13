from fastapi import FastAPI, HTTPException
from compiler import LLMCompiler
from pydantic import BaseModel

app = FastAPI()
compiler = LLMCompiler()

class Constraints(BaseModel):
    max_latency: float
    priority_speed: bool = True

@app.post("/recommend")
def get_recommendation(constraints: Constraints):
    weights = {'latency': 0.9, 'ppl': 0.1} if constraints.priority_speed else {'latency': 0.4, 'ppl': 0.6}
    
    best, status = compiler.compile(max_latency=constraints.max_latency, weights=weights)
    
    if best is None:
        raise HTTPException(status_code=404, detail=status)
        
    return {
        "recommended_model": best['Model'],
        "metrics": {
            "latency": float(best['Latency(s)']),
            "perplexity": float(best['Perplexity'])
        }
    }