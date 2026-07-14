import json

import pandas as pd

from arbitrar_divergencias import ARQUIVO_ARBITRAGEM, CHECKPOINT, MODELO_JUIZ

decisoes = {}
with open(CHECKPOINT, encoding="utf-8") as f:
    for linha in f:
        registro = json.loads(linha)
        decisoes[registro["url_base"]] = registro["valores"]

df = pd.read_excel(ARQUIVO_ARBITRAGEM)
df["valor_correto"] = df["valor_correto"].astype(object)
if "arbitrado_por" not in df.columns:
    df["arbitrado_por"] = None
df["arbitrado_por"] = df["arbitrado_por"].astype(object)

preenchidas = 0
for i, linha in df.iterrows():
    valores = decisoes.get(linha["url_base"], {})
    if pd.isna(linha["valor_correto"]) and linha["campo"] in valores:
        df.at[i, "valor_correto"] = valores[linha["campo"]]
        df.at[i, "arbitrado_por"] = MODELO_JUIZ
        preenchidas += 1

df.to_excel(ARQUIVO_ARBITRAGEM, index=False)
pendentes = df["valor_correto"].isna().sum()
print(f"{preenchidas} divergências preenchidas pelo juiz · {pendentes} pendentes")
