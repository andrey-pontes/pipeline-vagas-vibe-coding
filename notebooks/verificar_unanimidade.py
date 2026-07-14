import pandas as pd
from pydantic import BaseModel, ConfigDict

from arbitrar_divergencias import VALIDATION_DIR
from extracao_comum import PROMPT_SISTEMA, criar_cliente

MODELO_JUIZ = "anthropic/claude-sonnet-5"
ARQUIVO_VERIFICACAO = VALIDATION_DIR / "verificacao_unanimidade.xlsx"

CAMPOS = [
    "usa_ia_no_desenvolvimento", "senioridade", "modalidade",
    "exige_ingles", "exige_formacao_superior",
    "skills_tecnicas", "praticas_genai", "ferramentas_ia_codigo",
]

PROMPT_VERIFICACAO = PROMPT_SISTEMA + """

Você atua como auditor: três modelos extraíram esta vaga com as regras acima e concordaram em todos os campos. Releia a descrição e diga em quais campos a extração unânime está ERRADA — liste apenas os nomes desses campos, ou lista vazia se a extração inteira estiver correta. Nas listas (skills_tecnicas, praticas_genai, ferramentas_ia_codigo), considere errado apenas desvio material (item inventado ou omissão relevante), não diferenças de granularidade."""


class Auditoria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campos_errados: list[str]


FORMATO_AUDITORIA = {
    "type": "json_schema",
    "json_schema": {
        "name": "auditoria",
        "strict": True,
        "schema": Auditoria.model_json_schema(),
    },
}


def main():
    verificacao = pd.read_excel(ARQUIVO_VERIFICACAO)
    verificacao["campos_errados"] = verificacao["campos_errados"].astype(object)
    cliente = criar_cliente()

    for i, linha in verificacao.iterrows():
        if pd.notna(linha["campos_errados"]):
            continue
        extracao = "\n".join(f"- {campo}: {linha[campo]}" for campo in CAMPOS)
        resposta = cliente.chat.completions.create(
            model=MODELO_JUIZ,
            messages=[
                {"role": "system", "content": PROMPT_VERIFICACAO},
                {"role": "user", "content":
                    f"DESCRIÇÃO DA VAGA:\n{linha['descricao']}\n\n"
                    f"EXTRAÇÃO UNÂNIME:\n{extracao}"},
            ],
            response_format=FORMATO_AUDITORIA,
            extra_body={"provider": {"require_parameters": True}},
            timeout=180,
        )
        errados = Auditoria.model_validate_json(
            resposta.choices[0].message.content).campos_errados
        errados = [c.strip() for c in errados if c.strip() in CAMPOS]
        verificacao.at[i, "campos_errados"] = "; ".join(errados) if errados else "ok"
        print(f"[{i + 1}/{len(verificacao)}] {linha['job_title']}: "
              f"{verificacao.at[i, 'campos_errados']}")

    verificacao.to_excel(ARQUIVO_VERIFICACAO, index=False)
    print(f"salvo em {ARQUIVO_VERIFICACAO}")


if __name__ == "__main__":
    main()
