import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from extracao_comum import PROMPT_SISTEMA, criar_cliente

MODELO_JUIZ = "anthropic/claude-sonnet-5"
VALIDATION_DIR = Path(__file__).resolve().parent.parent / "data" / "validation"
ARQUIVO_ARBITRAGEM = VALIDATION_DIR / "divergencias_arbitragem.xlsx"
CHECKPOINT = VALIDATION_DIR / "arbitragem_juiz.jsonl"

MODELOS = [
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v4-flash",
    "google/gemma-4-31b-it",
]
VALORES_PERMITIDOS = {
    "usa_ia_no_desenvolvimento": ["exige", "valoriza", "menciona", "nao_menciona"],
    "senioridade": ["estagio", "junior", "pleno", "senior", "lider", "nao_informado"],
    "modalidade": ["remoto", "hibrido", "presencial", "nao_informado"],
    "exige_ingles": ["true", "false"],
    "exige_formacao_superior": ["true", "false"],
}

PROMPT_JUIZ = PROMPT_SISTEMA + """

Você atua como árbitro: três modelos extraíram esta vaga com as regras acima e divergiram nos campos listados. Releia a descrição e decida o valor correto de cada campo divergente. As respostas dos modelos são contexto, não voto — você pode discordar das três. Para skills_tecnicas, responda a lista completa correta separada por '; ' (as três listas ajudam a não omitir itens, mas inclua apenas o que a descrição sustenta). Nos demais campos, responda um dos valores permitidos, exatamente como escrito."""


class Decisao(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campo: str
    valor_correto: str


class Arbitragem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisoes: list[Decisao]


FORMATO_JUIZ = {
    "type": "json_schema",
    "json_schema": {
        "name": "arbitragem",
        "strict": True,
        "schema": Arbitragem.model_json_schema(),
    },
}


def montar_mensagem(grupo):
    partes = [f"DESCRIÇÃO DA VAGA:\n{grupo.iloc[0]['descricao']}\n\nCAMPOS DIVERGENTES:"]
    for _, linha in grupo.iterrows():
        campo = linha["campo"]
        partes.append(f"\n- campo: {campo}")
        if campo in VALORES_PERMITIDOS:
            partes.append(f"  valores permitidos: {', '.join(VALORES_PERMITIDOS[campo])}")
        for modelo in MODELOS:
            partes.append(f"  {modelo}: {linha[modelo]}")
    return "\n".join(partes)


def arbitrar_vaga(cliente, url, grupo, tentativas=3):
    campos_esperados = set(grupo["campo"])
    for tentativa in range(1, tentativas + 1):
        resposta = cliente.chat.completions.create(
            model=MODELO_JUIZ,
            messages=[
                {"role": "system", "content": PROMPT_JUIZ},
                {"role": "user", "content": montar_mensagem(grupo)},
            ],
            response_format=FORMATO_JUIZ,
            extra_body={"provider": {"require_parameters": True}},
            timeout=180,
        )
        decisoes = Arbitragem.model_validate_json(
            resposta.choices[0].message.content).decisoes
        valores = {d.campo: d.valor_correto.strip() for d in decisoes}
        validas = (
            set(valores) == campos_esperados
            and all(valores[c].lower() in VALORES_PERMITIDOS[c]
                    for c in valores if c in VALORES_PERMITIDOS)
        )
        if validas:
            return {c: (v.lower() if c in VALORES_PERMITIDOS else v)
                    for c, v in valores.items()}
        if tentativa == tentativas:
            raise ValueError(f"resposta inválida do juiz para {url}: {valores}")


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    divergencias = pd.read_excel(ARQUIVO_ARBITRAGEM)
    feitas = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            feitas = {json.loads(linha)["url_base"] for linha in f}

    grupos = {url: grupo for url, grupo in divergencias.groupby("url_base")
              if url not in feitas}
    print(f"{len(grupos)} vagas pendentes de {divergencias['url_base'].nunique()}")
    if smoke:
        grupos = dict(list(grupos.items())[:1])

    cliente = criar_cliente()
    with open(CHECKPOINT, "a", encoding="utf-8") as ckpt:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futuros = {pool.submit(arbitrar_vaga, cliente, url, grupo): url
                       for url, grupo in grupos.items()}
            for i, futuro in enumerate(as_completed(futuros), 1):
                url = futuros[futuro]
                try:
                    valores = futuro.result()
                except Exception as erro:
                    print(f"FALHA {url}: {repr(erro)[:150]}")
                    continue
                ckpt.write(json.dumps({"url_base": url, "valores": valores},
                                      ensure_ascii=False) + "\n")
                ckpt.flush()
                print(f"[{i}/{len(grupos)}] {url}: {valores}")


if __name__ == "__main__":
    main()
