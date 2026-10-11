# Quando esse deal vai fechar?

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isabelaclass/when-will-this-deal-close/blob/main/notebooks/quando-esse-deal-vai-fechar.ipynb)

Estudo de caso do portfólio de [Isabela Class](https://isabelaclass.com): prever **quando** um negócio fecha e transformar isso em previsão de receita mensal, testada contra o que de fato aconteceu. A versão para ler, com o simulador interativo, está no site; aqui está o código para conferir e reproduzir.

Aqui, **fechar é ganhar**: o negócio vira receita. Quando ele sai do funil, ganho ou perdido, digo que foi **encerrado**; o que nunca é ganho nem perdido fica **parado**.

Status: estudo em andamento.

## O que o estudo encontrou

- O tempo até o ganho tem dois grupos: 42% dos ganhos saem em até 20 dias e 54% levam de 45 a 138. A média, 52 dias, não descreve quase ninguém.
- De cada 100 negócios que entram em negociação, cerca de 51 são ganhos, 32 perdidos e 16 ficam parados para sempre. Depois de 138 dias aberto, nenhum foi ganho.
- O que decide o *quando* é a idade do negócio e o mês do trimestre. Produto, vendedor, regional e porte da conta não melhoraram a previsão.
- Em sete datas de corte, a previsão mensal (negócios abertos + negócios que ainda vão entrar) errou em média 13%. A conta tradicional, pipeline × taxa de ganho, previu três vezes o real.
- Um alarme simples encontrou uma regional que triplicou o pipeline em três semanas com negócios que ficaram parados; descontá-los reduziu o erro médio de 18% para 13%.

## Como rodar

```bash
pip install -r requirements.txt
```

1. Baixe a base **CRM Sales Opportunities** no [Data Playground da Maven Analytics](https://mavenanalytics.io/data-playground) e coloque os cinco CSVs em `data/` (detalhes em [`data/README.md`](data/README.md)).
2. Abra `notebooks/quando-esse-deal-vai-fechar.ipynb` e rode as células em ordem. No Colab, a primeira célula clona este repositório; depois é só enviar os CSVs para `data/`.

## Organização

| Caminho | O que tem |
|---|---|
| `notebooks/quando-esse-deal-vai-fechar.ipynb` | A análise inteira, na ordem da história, com as saídas gravadas |
| `src/deal/dados.py` | Leitura e limpeza das tabelas |
| `src/deal/modelos.py` | Modelos de risco em tempo discreto (idade; idade + calendário; gradient boosting) e o teste retroativo |
| `src/deal/alarme.py` | O detector de pipeline inflado |

## Método, em uma frase

Cada negócio aberto, a cada dia, pode ser ganho, perdido ou continuar aberto; a chance de cada destino vem do histórico de negócios com a mesma idade, no mesmo mês do trimestre. Avançando dia a dia e multiplicando pela receita esperada, sai a previsão do mês. O teste usa, em cada data de corte, só o que se sabia até ali.

## Limitações

- A base é pública e **sintética**; os padrões são do gerador dela. O método vale, os números são desta empresa fictícia.
- Os encerramentos só existem entre 01/03 e 31/12/2017, então a curva de ganhos usa apenas os negócios com 138 dias completos de observação.
- Não há histórico de atividades (ligações, reuniões, mudanças de etapa).
- O campo de conta vazia não é usado: todo negócio sem conta está aberto, então ele revelaria o resultado.
- Sete datas de corte são poucas para afirmar a precisão com segurança.

---

**English summary.** Portfolio case study: predicting *when* a B2B deal will close (here, closing means winning) and turning that into a monthly revenue forecast, backtested on seven cut-off dates. A discrete-time hazard model using only deal age and month-of-quarter beat the classic pipeline × win-rate forecast by a wide margin (13% vs. ~3x error), and a simple detector flagged a sales region that tripled its pipeline with deals that got stuck. Data: Maven Analytics' synthetic CRM Sales Opportunities dataset (not included; see `data/README.md`).
