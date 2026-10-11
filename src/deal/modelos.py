"""Modelos de risco em tempo discreto e teste retroativo.

Em uma data de corte D, prevê a receita ganha nos meses seguintes usando só o que se sabia até D."""
import numpy as np
import pandas as pd
from .dados import INI, dia_tri, mes_tri
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

def age_bin(k):
    k=np.asarray(k); return np.where(k<=20,k,20+np.ceil((k-20)/5)).astype(int)

def pessoa_dia(e,D):
    """Uma linha por negocio por dia em risco, ate D. Respeita o corte de 1/3/2017 (antes disso a base nao registra encerramentos)."""
    d=e[e.engage_date<D].copy()
    encerrado=d.close_date.notna()&(d.close_date<=D)
    ult=np.where(encerrado,d.t,(D-d.engage_date).dt.days).astype(int)       # ultimo dia observado
    prim=np.maximum(1,(INI-d.engage_date).dt.days.values).astype(int)     # primeiro dia observavel
    n=np.maximum(0,ult-prim+1)
    idx=np.repeat(np.arange(len(d)),n)
    k=np.concatenate([np.arange(a,b+1) for a,b in zip(prim,ult) if b>=a]) if n.sum() else np.array([],int)
    pd_=pd.DataFrame({'i':d.index.values[idx],'k':k})
    pd_['data']=d.engage_date.values[idx]+pd.to_timedelta(k,'D')
    evk=np.where(encerrado.values[idx]&(k==ult[idx]),d.ev.values[idx],0)
    pd_['y']=evk
    pd_['dq']=dia_tri(pd_.data); pd_['mq']=mes_tri(pd_.data)
    pd_['dq_ent']=dia_tri(pd.Series(d.engage_date.values[idx])).values
    pd_['preco']=np.log(d.sales_price.values[idx])
    return pd_

class Naive:  # pipeline x taxa de ganho historica, tudo cai no periodo
    nome='M0 taxa de ganho'
class Idade:
    nome='M1 idade'
    def fit(self,pdm):
        g=pdm.assign(b=age_bin(pdm.k)).groupby('b').y.agg(n='size',w=lambda s:(s==1).sum(),l=lambda s:(s==2).sum())
        self.hw=(g.w/g.n).to_dict(); self.hl=(g.l/g.n).to_dict(); return self
    def haz(self,k,data,dq_ent,preco):
        b=age_bin(k); return np.array([self.hw.get(x,0.) for x in b]),np.array([self.hl.get(x,0.) for x in b])
class IdadeCal:
    nome='M2 idade + calendario'
    def fit(self,pdm):
        g=pdm.assign(b=age_bin(pdm.k)).groupby(['b','mq']).y.agg(n='size',w=lambda s:(s==1).sum(),l=lambda s:(s==2).sum())
        self.hw=(g.w/g.n).to_dict(); self.hl=(g.l/g.n).to_dict(); return self
    def haz(self,k,data,dq_ent,preco):
        b=age_bin(k); m=mes_tri(pd.Series(data)).values
        return np.array([self.hw.get((x,y),0.) for x,y in zip(b,m)]),np.array([self.hl.get((x,y),0.) for x,y in zip(b,m)])
class GBM:
    nome='M3 aprendizado de maquina'
    def __init__(self,feats=('k','dq','dq_ent')): self.feats=list(feats)
    def fit(self,pdm):
        self.m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,max_leaf_nodes=31,min_samples_leaf=200,random_state=0).fit(pdm[self.feats],pdm.y)
        self.kmax=pdm.k[pdm.y>0].max(); return self
    def haz(self,k,data,dq_ent,preco):
        X=pd.DataFrame({'k':k,'dq':dia_tri(pd.Series(data)).values,'dq_ent':dq_ent,'preco':preco})[self.feats]
        pr=self.m.predict_proba(X); cl=list(self.m.classes_)
        hw=pr[:,cl.index(1)]; hl=pr[:,cl.index(2)]
        z=np.asarray(k)>self.kmax; hw[z]=0; hl[z]=0   # depois da maior idade em que algum negocio foi encerrado, risco zero
        return hw,hl

def projetar(modelo,eng,preco,D,fim):
    """Para negocios com data de entrada eng, abertos em D: prob. de ganhar em cada dia de D+1 ate fim. Retorna matriz (negocio x dia)."""
    H=(fim-D).days; n=len(eng); a=(D-eng).days.values if hasattr((D-eng),'days') else (D-eng).dt.days.values
    a=np.maximum(a,0); ini_d=np.maximum(0,(eng-D).dt.days.values)   # negocios futuros so entram em risco depois da entrada
    dqe=dia_tri(pd.Series(eng.values)).values
    S=np.ones(n); Pw=np.zeros((n,H)); Pl=np.zeros((n,H))
    for d in range(1,H+1):
        data=D+pd.Timedelta(days=d); k=(data-eng).dt.days.values
        ativo=k>=1
        hw,hl=modelo.haz(np.where(ativo,k,1),np.repeat(np.datetime64(data),n),dqe,preco)
        hw=np.where(ativo,hw,0); hl=np.where(ativo,hl,0)
        Pw[:,d-1]=S*hw; Pl[:,d-1]=S*hl; S=S*(1-hw-hl)
    return Pw,Pl

def rodar(e,D,fim,modelos,verbose=True):
    D=pd.Timestamp(D); fim=pd.Timestamp(fim)
    pdm=pessoa_dia(e,D)
    hist=e[e.close_date.notna()&(e.close_date<=D)]
    razao=(hist[hist.ev==1].close_value/hist[hist.ev==1].sales_price).mean(); wr=(hist.ev==1).mean()
    ab=e[(e.engage_date<D)&~(e.close_date<=D)].copy()                      # pipeline aberto em D
    novos=e[(e.engage_date>=D)&(e.engage_date<=fim)].copy()               # entraram depois de D (desconhecidos em D)
    dias=pd.date_range(D+pd.Timedelta(days=1),fim); meses=dias.to_period('M')
    def real(d):
        g=d[(d.ev==1)&(d.close_date>D)&(d.close_date<=fim)]; return g.groupby(g.close_date.dt.to_period('M')).close_value.sum().reindex(meses.unique(),fill_value=0)
    out={'D':D,'n_abertos':len(ab),'valor_tabela_abertos':ab.sales_price.sum(),'real_pipeline':real(ab),'real_novos':real(novos),'wr_hist':wr,'razao':razao}
    # fluxo futuro de entradas: media diaria dos ultimos 90 dias, com o preco medio do mesmo periodo
    rec=e[(e.engage_date>=D-pd.Timedelta(days=90))&(e.engage_date<D)]
    taxa=len(rec)/90; preco_med=rec.sales_price.mean()
    fut_eng=pd.Series(np.repeat(dias.values,1)); out['entradas_dia']=taxa
    prev={}; nivel={}
    for M in modelos:
        if isinstance(M,Naive):
            prev[M.nome]={'pipeline_total':ab.sales_price.sum()*razao*wr}; continue
        M.fit(pdm)
        Pw,Pl=projetar(M,ab.engage_date.reset_index(drop=True),np.log(ab.sales_price.values),D,fim)
        v=ab.sales_price.values*razao
        pip=pd.Series((Pw*v[:,None]).sum(0),index=meses).groupby(level=0).sum()
        Pwf,_=projetar(M,pd.Series(dias.values),np.repeat(np.log(preco_med),len(dias)),D,fim)
        nov=pd.Series((Pwf*taxa*preco_med*razao).sum(0),index=meses).groupby(level=0).sum()
        prev[M.nome]={'pipeline':pip,'novos':nov}
        y=((ab.ev==1)&(ab.close_date<=fim)).astype(int).values; yc=(ab.close_date<=fim).astype(int).values
        nivel[M.nome]={'p_won':Pw.sum(1),'p_close':Pw.sum(1)+Pl.sum(1),'y_won':y,'y_close':yc,'Pw':Pw,'ab':ab}
    out['prev']=prev; out['nivel']=nivel
    return out
