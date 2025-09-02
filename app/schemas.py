from pydantic import BaseModel, Field

class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)

class AnswerResponse(BaseModel):
    answer: str
    strategy: str
