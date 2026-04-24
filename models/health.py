from pydantic import BaseModel
 
 
class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_reachable: bool
 