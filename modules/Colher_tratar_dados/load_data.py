import pandas as pd
import numpy as np
from datetime import datetime
import os
from statsmodels.tsa.arima.model import ARIMA
import warnings
# Suprimir temporariamente os avisos
warnings.filterwarnings("ignore")

def load_data(data_simulacao):
    ## Atribuir uma data inicial para a simulação, alguns dados antigos estão em desacordo
    data_inicial = datetime(2006,1,1)

    ## Pegar os ativos elegíveis
    arquivo_Ativos_Elegiveis = os.path.join("dataset", "BR", "ACOES", "IBOV_Elegivel.parquet")
    df_Elegivel = pd.read_parquet(arquivo_Ativos_Elegiveis)
    condicao_selecao_data = (df_Elegivel.index <= data_simulacao) & (df_Elegivel.index >= data_inicial)
    data_dos_Elegiveis = df_Elegivel.loc[condicao_selecao_data,:].index.max()
    condicao_selecao_ativos = df_Elegivel.loc[data_dos_Elegiveis,:].values == 1
    lista_ativos_elegiveis = df_Elegivel.loc[data_dos_Elegiveis,condicao_selecao_ativos].index.to_list()

 
    ## Dados da Selic
    arquivo_Selic = os.path.join("dataset", "BR", "Selic","Selic.parquet")

    df_Selic = pd.read_parquet(arquivo_Selic)
    # Coluna data como índice
    df_Selic.set_index('data', inplace=True)
    df_Selic.index = pd.to_datetime(df_Selic.index, format='%d/%m/%Y')

    # Lista de nomes de colunas que devem ser lidas como números
    colunas_numericas = ['anula100', 'diario'] 
    df_Selic.fillna(0,inplace=True)
    for col in colunas_numericas:
        # Substitua vírgulas por pontos e converta para float
        try:
            df_Selic[col] = df_Selic[col].str.replace(',', '.').astype(float)
        except:
            print(f"Erro na coluna {col}")
        pass
        

    # Filtrar a data de simulação
    condicao_selecao_data = (df_Selic.index <= data_simulacao) & (df_Selic.index >= data_inicial)
    df_Selic = df_Selic.loc[condicao_selecao_data,:]

    ## (FIM) Dados da Selic

    ## Dados da Expectativa de Selic
    arquivo_Expectativa_Selic = os.path.join("dataset","BR", "Selic", "Expectativa_Selic_Diaria.parquet")
    df_Expectativa_Selic = pd.read_parquet(arquivo_Expectativa_Selic)

    # Filtrar a data
    condicao_selecao_data = (df_Expectativa_Selic.index <= data_simulacao) & (df_Expectativa_Selic.index >= data_inicial)
    df_Expectativa_Selic = df_Expectativa_Selic.loc[condicao_selecao_data,:]

    # Média Mensal
    df_Expectativa_Selic_mensal = df_Expectativa_Selic.resample('M').mean()

    ## (FIM) Dados da Expectativa de Selic

    ## Dados das ações
    dict_df_acoes = {}
    Lista_nao_encontrados = []
    for Ticker in lista_ativos_elegiveis:
        try:
           ## Ler os dados normalizados
            arquivo_por_acao = os.path.join("dataset", "BR", "ACOES", "Dados_Tratados", "Dados_normalizados_acao_"+Ticker + ".parquet")
            df_acao = pd.read_parquet(arquivo_por_acao)
            ## Filtrar a data
            condicao_selecao_data = (df_acao["Data_balanco"] <= data_simulacao) & (df_acao["Data_balanco"] >= data_inicial)
            df_acao = df_acao.loc[condicao_selecao_data,:]
        
            ## Ler os dados dos múltiplos
            arquivo_por_multiplos = os.path.join("dataset", "BR", "ACOES", "Dados_Tratados", "Multiplos_diarios_"+Ticker + ".parquet")
            df_multiplos = pd.read_parquet(arquivo_por_multiplos)
            ## Filtrar a data
            condicao_selecao_data = (df_multiplos.index <= data_simulacao) & (df_multiplos.index >= data_inicial)
            df_multiplos = df_multiplos.loc[condicao_selecao_data,:]

            ## Ler o CAGR
            arquivo_por_CAGR = os.path.join("dataset", "BR", "ACOES", "Dados_Tratados", "CAGR_"+Ticker + ".parquet")
            df_CAGR = pd.read_parquet(arquivo_por_CAGR)
            ## Filtrar a data
            condicao_selecao_data = (df_CAGR.index <= data_simulacao) & (df_CAGR.index >= data_inicial)
            df_CAGR = df_CAGR.loc[condicao_selecao_data,:]

            ## Salvando os dados no dicionário
            dict_df_acoes[Ticker] = [df_acao, df_multiplos, df_CAGR]
        except:
            Lista_nao_encontrados.append(Ticker)
            pass


    
    return [dict_df_acoes, df_Selic, df_Expectativa_Selic_mensal]


def load_data_backtest(data_simulacao, lista_ativos_elegiveis, meses=12):
    df_cotacao_ajustado = pd.DataFrame()
    for ativo in lista_ativos_elegiveis:
        ## Ler os dados dos múltiplos
        arquivo_por_multiplos = os.path.join("dataset", "BR", "ACOES", "Dados_Tratados", "Multiplos_diarios_"+Ticker + ".parquet")
        df_multiplos = pd.read_parquet(arquivo_por_multiplos)

        ## Filtrar a data
        data_x_meses_afrente = data_simulacao + pd.DateOffset(months=meses)
        condicao_selecao_data = \
            (df_multiplos.index <= data_x_meses_afrente) & (df_multiplos.index >= data_simulacao)

        df_multiplos = df_multiplos.loc[condicao_selecao_data,:]
        ## Adicionando a cotação ajustada
        df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
        df_ativo.columns = [ativo]
        df_cotacao_ajustado = pd.concat([df_cotacao_ajustado, df_ativo], axis=1, join="outer")

    return df_cotacao_ajustado
