"""Leitura e limpeza da base Maven CRM Sales Opportunities."""
from pathlib import Path
import pandas as pd

PASTA_DADOS = Path(__file__).resolve().parents[2] / "data"
INI = pd.Timestamp("2017-03-01")   # primeiro fechamento registrado na base
FIM = pd.Timestamp("2017-12-31")   # último fechamento registrado na base


def ler_tabelas(pasta=PASTA_DADOS):
    pasta = Path(pasta)
    return {
        "pipeline": pd.read_csv(pasta / "sales_pipeline.csv", parse_dates=["engage_date", "close_date"]),
        "accounts": pd.read_csv(pasta / "accounts.csv"),
        "products": pd.read_csv(pasta / "products.csv"),
        "teams": pd.read_csv(pasta / "sales_teams.csv"),
    }


def limpar(t):
    """Corrige o que a investigação das colunas encontrou (ver notebook, seção 1)."""
    t = {k: v.copy() for k, v in t.items()}
    t["pipeline"]["product"] = t["pipeline"]["product"].replace({"GTXPro": "GTX Pro"})
    t["accounts"]["sector"] = t["accounts"]["sector"].replace({"technolgy": "technology"})
    t["accounts"]["office_location"] = t["accounts"]["office_location"].replace({"Philipines": "Philippines"})
    return t


def carregar(pasta=PASTA_DADOS):
    """Negócios que já entraram em Engaging (ganhos, perdidos e abertos), com produto e time."""
    t = limpar(ler_tabelas(pasta))
    p = t["pipeline"].merge(t["products"], on="product", how="left").merge(t["teams"], on="sales_agent", how="left")
    e = p[p.deal_stage != "Prospecting"].copy().reset_index(drop=True)
    e["ev"] = e.deal_stage.map({"Won": 1, "Lost": 2, "Engaging": 0})   # 1 ganho, 2 perdido, 0 aberto
    e["t"] = (e.close_date - e.engage_date).dt.days                    # dias até fechar (vazio se aberto)
    return e


def dia_tri(d):
    """Dia dentro do trimestre (1 a 92)."""
    d = pd.to_datetime(d)
    if hasattr(d, "dt"):
        return (d - d.dt.to_period("Q").dt.start_time).dt.days + 1
    return (d - d.to_period("Q").start_time).days + 1


def mes_tri(d):
    """Mês dentro do trimestre (1, 2 ou 3)."""
    d = pd.to_datetime(d)
    return ((d.dt.month - 1) % 3 + 1) if hasattr(d, "dt") else (d.month - 1) % 3 + 1
