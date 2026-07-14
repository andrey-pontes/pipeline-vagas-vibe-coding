
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


class ExtracaoVaga(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills_tecnicas: list[str] = Field(
        description="Tecnologias, linguagens, frameworks e plataformas exigidas ou desejadas, "
                    "normalizadas em inglês minúsculo (ex.: python, machine learning, aws, docker)"
    )
    praticas_genai: list[str] = Field(
        description="Práticas de IA generativa citadas na vaga, em minúsculo "
                    "(ex.: llm, rag, prompt engineering, ai agents, fine-tuning); vazio se nenhuma"
    )
    ferramentas_ia_codigo: list[str] = Field(
        description="Ferramentas de IA para apoiar a escrita de código citadas "
                    "(ex.: github copilot, cursor, claude code, windsurf); vazio se nenhuma"
    )
    usa_ia_no_desenvolvimento: Literal["exige", "valoriza", "menciona", "nao_menciona"] = Field(
        description="Se a vaga espera que a pessoa use IA no próprio fluxo de trabalho de "
                    "desenvolvimento: 'exige' (requisito), 'valoriza' (diferencial), "
                    "'menciona' (cita sem exigir) ou 'nao_menciona'"
    )
    senioridade: Literal["estagio", "junior", "pleno", "senior", "lider", "nao_informado"] = Field(
        description="Senioridade indicada no título ou na descrição; 'lider' cobre tech lead, "
                    "coordenação e gestão"
    )
    modalidade: Literal["remoto", "hibrido", "presencial", "nao_informado"] = Field(
        description="Modalidade de trabalho declarada na descrição"
    )
    exige_ingles: bool = Field(
        description="True se a vaga exige inglês ou se a descrição inteira é em inglês para "
                    "empresa internacional"
    )
    exige_formacao_superior: bool = Field(
        description="True se a vaga exige graduação completa ou em andamento"
    )


PROMPT_SISTEMA = """Você extrai informações estruturadas de vagas de emprego de tecnologia.
Analise a descrição da vaga (em português ou inglês) e preencha o schema solicitado.
Normalize as skills para inglês em minúsculas, unificando traduções e variações:
"aprendizado de máquina", "ML" e "machine learning" viram "machine learning";
"bancos de dados relacionais" vira "sql". Use o nome mais comum de cada tecnologia.
Conceitos de IA generativa pertencem a praticas_genai, não a skills_tecnicas; use nomes canônicos ("large language models"/"LLMs" viram "llm"; "agentes"/"agents" viram "ai agents") e registre somente práticas que o texto de fato cita.
Em senioridade e modalidade, responda 'nao_informado' quando a descrição não
declarar — não deduza do tom do texto.
Em usa_ia_no_desenvolvimento, conte qualquer expectativa de que a pessoa use ou construa com IA no próprio trabalho: 'exige' se for parte central do papel, 'valoriza' se for diferencial.
Registre apenas o que estiver explícito ou claramente implícito no texto da vaga."""

FORMATO_RESPOSTA = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracao_vaga",
        "strict": True,
        "schema": ExtracaoVaga.model_json_schema(),
    },
}


def criar_cliente():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def extrair_vaga(cliente, modelo, descricao, tentativas=3):
    for tentativa in range(1, tentativas + 1):
        try:
            resposta = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": descricao},
                ],
                response_format=FORMATO_RESPOSTA,
                extra_body={"provider": {"require_parameters": True}},
                timeout=120,
            )
            return ExtracaoVaga.model_validate_json(resposta.choices[0].message.content)
        except Exception:
            if tentativa == tentativas:
                raise
            time.sleep(5 * tentativa)
