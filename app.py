import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).parent
app = FastAPI(title="Inglês Brabo")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class ExplainRequest(BaseModel):
    image_data_url: str | None = None
    question: str = Field(default="Explique o inglês que aparece nesta tela.", max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


SYSTEM_PROMPT = """Você é o Inglês Brabo, um professor paciente para um brasileiro.
Explique SOMENTE o inglês que for legível na imagem ou que o aluno perguntar.
Não invente texto que não esteja visível. Se a tela não tiver uma frase inglesa legível,
peça uma imagem melhor ou que ele cole a frase.

Escreva em português brasileiro simples e use este formato:
## Frase e sentido natural
Mostre a frase em inglês e uma tradução natural (não palavra por palavra).
## Palavras e expressões
Explique as partes importantes no contexto, incluindo phrasal verbs e contrações.
## Como a frase foi montada
Identifique sujeito, verbo, objeto/complementos e a regra gramatical relevante.
## Por que a ordem muda em português
Compare a ordem inglesa com a ordem natural em português, sem dizer que uma tradução literal é obrigatória.
## Treino rápido
Crie uma pergunta curta para o aluno responder ou uma variação da frase.

Se ele fizer uma pergunta sobre uma explicação anterior, responda diretamente e mantenha exemplos curtos."""


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/api/explain")
def explain(payload: ExplainRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "Configure OPENAI_API_KEY no arquivo .env antes de analisar a tela.")

    content = [{"type": "input_text", "text": payload.question}]
    if payload.image_data_url:
        content.append({"type": "input_image", "image_url": payload.image_data_url, "detail": "low"})

    previous = []
    for item in payload.history[-8:]:
        role = "assistant" if item.get("role") == "assistant" else "user"
        text = item.get("text", "")[:4000]
        if text:
            previous.append({"role": role, "content": [{"type": "input_text", "text": text}]})

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions=SYSTEM_PROMPT,
            input=previous + [{"role": "user", "content": content}],
            max_output_tokens=1100,
        )
        return {"answer": response.output_text}
    except Exception as error:
        raise HTTPException(502, f"Não foi possível consultar o professor: {error}") from error

