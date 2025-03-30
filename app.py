
from fastapi import FastAPI
from .schemas import TextSchema, TextAnalysis
import services

app = FastAPI()
BASE_URL = "api/v1"


@app.post(path=f"{BASE_URL}/verify", response_model=TextAnalysis)
async def verify(text: TextSchema):
    text_analysis = services.analyse_text(text.text)
    return text_analysis


@app.post(path=f"{BASE_URL}/cross-validate")
async def cross_validate(text: TextSchema):
    result = services.cross_validate_news(text.text)
