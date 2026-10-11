"""Detector de pipeline inflado: compara, semana a semana, a fracao de negocios encerrados (ganhos ou perdidos) em ate 20 dias
em cada regional com a das outras regionais na mesma semana. Usa so o que se sabe na data de corte D."""
import numpy as np
import pandas as pd
from .dados import INI
from .modelos import rodar, IdadeCal
from scipy.stats import binomtest
RAP=20
def detectar(e,D,alpha=0.01,min_pico=1.5,verbose=False):
    D=pd.Timestamp(D)
    d=e[(e.engage_date>=INI)&(e.engage_date<=D-pd.Timedelta(days=RAP))].copy()
    d['wk']=d.engage_date.dt.to_period('W'); d['rap']=(d.close_date<=D)&(d.t<=RAP)
    g=d.groupby(['wk','regional_office']).agg(N=('rap','size'),F=('rap','sum')).reset_index()
    tot=g.groupby('wk')[['N','F']].transform('sum')
    g['f_outros']=(tot.F-g.F)/(tot.N-g.N)
    g['p']=[binomtest(int(F),int(N),f,alternative='less').pvalue if f>0.05 else 1 for F,N,f in zip(g.F,g.N,g.f_outros)]
    g['N_vivo']=np.where(g.f_outros>0.05,g.F/g.f_outros,g.N)
    # segunda condicao: volume da semana bem acima do normal da propria regional (mediana das 8 semanas anteriores)
    g=g.sort_values(['regional_office','wk'])
    g['N_normal']=g.groupby('regional_office').N.transform(lambda s:s.shift(1).rolling(8,min_periods=4).median())
    g['pico']=g.N/g.N_normal
    g['excesso']=np.where((g.p<alpha)&(g.pico>=min_pico),np.maximum(0,g.N-g.N_vivo),0)
    ab=e[(e.engage_date<D)&~(e.close_date<=D)].copy(); ab['wk']=ab.engage_date.dt.to_period('W')
    O=ab.groupby(['wk','regional_office']).size().rename('O').reset_index()
    g=g.merge(O,on=['wk','regional_office'],how='left').fillna({'O':0})
    g['fator']=np.where(g.excesso>0,np.clip((g.O-g.excesso)/g.O.replace(0,np.nan),0,1),1.0)
    if verbose: print(g[(g.excesso>0)|(g.p<alpha)][['wk','regional_office','N','N_normal','pico','F','f_outros','p','N_vivo','excesso','O','fator']].to_string(float_format=lambda x:f'{x:,.2f}'))
    f=ab.merge(g[['wk','regional_office','fator']],on=['wk','regional_office'],how='left').fator.fillna(1.0).values
    return f,g

def rodar_ajustado(e,D,fim):
    r=rodar(e,D,fim,[IdadeCal()]); nv=r['nivel']['M2 idade + calendario']; ab=nv['ab']
    fator,g=detectar(e,D)
    v=ab.sales_price.values*r['razao']; meses=pd.date_range(pd.Timestamp(D)+pd.Timedelta(days=1),fim).to_period('M')
    Pw=nv['Pw']
    base=pd.Series((Pw*v[:,None]).sum(0),index=meses).groupby(level=0).sum()
    aj=pd.Series((Pw*(v*fator)[:,None]).sum(0),index=meses).groupby(level=0).sum()
    return r,base,aj,fator,g
