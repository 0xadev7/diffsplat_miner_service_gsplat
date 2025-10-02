from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    seed: Optional[int] = Field(None, description="Random seed (optional)")

class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    seed: Optional[int] = Field(None, description="Random seed (optional)")

class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    model_variant: str