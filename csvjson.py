import pandas as pd
import json

df = pd.read_csv("categorias.csv")

result = {}

for cat, subcat, item in df.itertuples(index=False):
    result.setdefault(cat, {}).setdefault(subcat, []).append(item)

with open("categorias.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)