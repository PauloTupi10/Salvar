# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os

# %% [markdown]
# # Definindo as funções

# %% [markdown]
# ## Funções Básicas

# %%
def endereco_arquivos(Ticker):
    ## Arquivos de Dados
    # Dados Fundamentalistas
    ticker_Fund = f"{Ticker}_Fund.parquet"
    arquivo_Fund = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Brutos",ticker_Fund)
    # Cotações Diárias
    ticker_Cot = f"{Ticker}_Cot.parquet"
    arquivo_Cot = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Brutos",ticker_Cot)
    # Dados de Proventos
    ticker_Prov = f"{Ticker}_Prov.parquet"
    arquivo_Prov = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Brutos",ticker_Prov)
    # Dados de Eventos como desdobramentos e grupamentos
    ticker_Eventos = f"{Ticker}_Eventos.parquet"
    arquivo_Eventos = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Brutos",ticker_Eventos)
    # Dados de Subscrições
    ticker_Subscricao = f"{Ticker}_Subscricao.parquet"
    arquivo_Subscricao = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Brutos",ticker_Subscricao)

    return arquivo_Fund, arquivo_Cot, arquivo_Prov, arquivo_Eventos, arquivo_Subscricao

# %%
def Func_Classe_acao(Ticker):
    if Ticker[4] == "3":
        Classe_acao = "ON"
    elif Ticker[4] == "4":
        Classe_acao = "PN"
    elif Ticker[4] == "5":
        Classe_acao = "PNA"
    elif Ticker[4] == "6":
        Classe_acao = "PNB"
    elif Ticker[4:6] == "11":
        Classe_acao = "UNT"
    else:
        Classe_acao = "ERRO"
        
    return Classe_acao

# %% [markdown]
# ## Tratar dados

# %%
def tratar_Fund(arquivo_Fund):
    # Lendo os arquivos
    df_fund = pd.read_parquet(arquivo_Fund)


    # Ao ler o arquivo Parquet, especifique o dtype usando o parâmetro dtype
    df_fund = pd.read_parquet(arquivo_Fund)

    # Excluir o que não tem Data_balanco
    df_fund = df_fund.dropna(subset=["Data_balanco","Num_acoes","Market_value"])

    # Colunas que são datas
    colunas_datas = ['Data_balanco', 'Data_demonstracao', 'Data_analise']  # Substitua pelos nomes reais das suas colunas de datas
    df_fund.loc[:, colunas_datas] = df_fund.loc[:, colunas_datas].apply(pd.to_datetime, format='%d/%m/%Y')
    df_fund.index = pd.to_datetime(df_fund.index, format='%d/%m/%Y')

    # Suponha que você tenha uma lista de nomes de colunas que devem ser lidas como números
    colunas_numericas = ['Num_acoes', 'Fator_equivalencia_acoes',
        'Market_value', 'PL', 'RL', 'EBITDA', 'D&A', 'EBIT', 'LL',
        'LL_controlador', 'LL_nao_controlador', 'ROIC', 'ROE', 'Div_Bruta',
        'Div_liq', 'Div_Arrendamento', 'FCO', 'FCI', 'FCF', 'Preco_fechamento',
        'Payout', 'Proventos', 'JCP',
        'DY_12m', "DY_24m", "DY_36m", "DY_48m", "DY_60m",
        'ret_12meses', 'ret_1mes_aa', 'ret_ano', 'ret_CDI_1m',
        'ret_CDI_12m', 'ret_CDI_ano', 'ret_IBOV_1mes', 'ret_IBOV_12m',
        'ret_IBOV_ano',
        'meses']  # Substitua pelos nomes reais das suas colunas numéricas
    df_fund.fillna(0,inplace=True)
    for col in colunas_numericas:
        # Substitua vírgulas por pontos e converta para float
        try:
            df_fund[col] = df_fund[col].str.replace(',', '.').astype(float)
        except:
            print(f"Erro na coluna {col}")
        pass

    df_fund.fillna(0,inplace=True)



    # Colunas que são porcentagem
    colunas_pct = ['DY_12m', "DY_24m", "DY_36m", "DY_48m", "DY_60m",
                   'ret_12meses', 'ret_1mes_aa', 'ret_ano', 'ret_CDI_1m',
        'ret_CDI_12m', 'ret_CDI_ano', 'ret_IBOV_1mes', 'ret_IBOV_12m',
        'ret_IBOV_ano']  # Substitua pelos nomes reais das suas colunas de porcentagem

    df_fund.loc[:, colunas_pct] = df_fund.loc[:, colunas_pct].apply(lambda x: x/100)

    df_fund.sort_index(ascending=False, inplace=True)
    return df_fund

# %%
def tratar_cot(arquivo_Cot):
    # Lendo os arquivos de cotações
    df_cot = pd.read_parquet(arquivo_Cot)
    # Colunas que são datas
    df_cot.index = pd.to_datetime(df_cot.index, format='%d/%m/%Y')

    # Substitua os valores vazios ("" ou string vazia) por NaN em todas as colunas
    df_cot.replace("", np.nan, inplace=True)

    # Excluir o que não tem Fech_Historico
    df_cot = df_cot.dropna(subset=["Fech_Historico"])


    # Colunas que são numéricas
    colunas_numericas = ['Fech_Ajustado', 'Variação(%)','Fech_Historico', 'Abertura_Ajustado',
        'Min_Ajustado', 'Medio_Ajustado', 'Max_Ajustado', 'Vol(MM_R$)',
        'Negocios', 'Fator']
    for col in colunas_numericas:
        # Substitua vírgulas por pontos e converta para float
        try:
            df_cot[col] = df_cot[col].str.replace(',', '.').astype(float)
        except:
            raise


    # Colunas que são porcentagem
    colunas_pct = ['Variação(%)']
    # Substitua os valores None por NaN em todas as colunas
    df_cot.fillna(0, inplace=True)
    df_cot.loc[:, colunas_pct] = df_cot.loc[:, colunas_pct].apply(lambda x: x/100)

    # Ordenar por index
    df_cot.sort_index(ascending=False, inplace=True)

    return df_cot

# %%
def tratar_prov(arquivo_Prov, Classe_acao):
    # Lendo os arquivos de proventos
    try:
        Existe_prov = True
        df_prov = pd.read_parquet(arquivo_Prov)

        # Colunas que são datas
        df_prov.index = pd.to_datetime(df_prov.index, format='%d/%m/%Y')

        # Substitua os valores vazios ("" ou string vazia) por NaN em todas as colunas
        df_prov.replace("", np.nan, inplace=True)

        # Excluir o que não tem Valor_do_Provento
        df_prov = df_prov.dropna(subset=["Valor_do_Provento"])

        # Colunas que são numéricas
        colunas_numericas = ['Valor_do_Provento', 'Último_preco_com', 'Provento_por']
        # Ordenar por index
        df_prov.sort_index(ascending=False, inplace=True)
        # Filtra a classe da ação
        df_prov = df_prov.loc[(df_prov["Tipo"]==Classe_acao) | (df_prov["Tipo"]=="todas"),:].copy()
        
        for col in colunas_numericas:
            # Substitua vírgulas por pontos e converta para float
            try:
                df_prov[col] = df_prov[col].str.replace(',', '.').astype(float)
            except:
                raise
 
    except:
        Existe_prov = False
        df_prov = pd.DataFrame()

    return df_prov, Existe_prov

# %%
def tratar_even(arquivo_Eventos, Classe_acao):
    # Lendo os arquivos de eventos
    try:
        Existe_eventos = True
        df_eventos = pd.read_parquet(arquivo_Eventos)

        # Colunas que são datas
        df_eventos.index = pd.to_datetime(df_eventos.index, format='%d/%m/%Y')
        colunas_numericas = ['Fator']
        # Filtra a classe da ação
        df_eventos = df_eventos.loc[(df_eventos["ClasseAcao"]==Classe_acao) | (df_eventos["ClasseAcao"]=="todas"),:].copy()
        # Ordenar por index
        df_eventos.sort_index(ascending=False, inplace=True)
        # Colunas que são numéricas
        for col in colunas_numericas:
            # Substitua vírgulas por pontos e converta para float
            try:
                df_eventos[col] = df_eventos[col].str.replace(',', '.').astype(float)
            except:
                raise

    except:
        Existe_eventos = False
        df_eventos = pd.DataFrame()
    

    return df_eventos, Existe_eventos

# %%
def tratar_subs(arquivo_Subscricao):
    # Lendo os arquivos de subscrições
    try:
        Existe_subscricao = True
        df_subscricao = pd.read_parquet(arquivo_Subscricao)
        # Colunas que são datas
        df_subscricao.index = pd.to_datetime(df_subscricao.index, format='%d/%m/%Y')

        # Substitua os valores vazios ("" ou string vazia) por NaN em todas as colunas
        df_subscricao.replace("", np.nan, inplace=True)

        # Excluir o que não tem Valor_do_Provento
        df_subscricao = df_subscricao.dropna(subset=["Fator"])

        # Colunas que são numéricas
        colunas_numericas = ['Fator', 'Preco_Subscricao']

        # Ordenar por index
        df_subscricao.sort_index(ascending=False, inplace=True)
        # Filtra a classe da ação
        df_subscricao = df_subscricao.loc[(df_subscricao["ClasseAcao"]==Classe_acao) | (df_subscricao["ClasseAcao"]=="todas"),:].copy()

        for col in colunas_numericas:
        # Substitua vírgulas por pontos e converta para float
            try:
                df_subscricao[col] = df_subscricao[col].str.replace(',', '.').astype(float)
            except:
                raise
        
    except:
        Existe_subscricao = False
        df_subscricao = pd.DataFrame()



    return df_subscricao, Existe_subscricao

# %% [markdown]
# ## Normalização

# %%
def normalizar_dados_fund(df_fund, df_eventos, Existe_eventos, Ticker):
    # Normalizar Dados Fundamentalistas e considerar os eventos
    df_Tratar_por_Acao = df_fund.copy()

    ## Calcular o número de ações Equivalentes atual
    # Criar a Coluna de Número de Ações Equivalentes
    df_Tratar_por_Acao.insert(3, "Num_acoes_equivalentes", 0)

    ## No caso de Units é necessário considerar o número de ações de cada classe
    df_Tratar_por_Acao.loc[:, "Num_acoes_equivalentes"]= df_Tratar_por_Acao.apply(lambda x: x["Num_acoes"] / x["Fator_equivalencia_acoes"], axis=1)



    ## Adicionar o Ticker
    df_Tratar_por_Acao.insert(0, "Ticker", Ticker)

    ## Adicionar o Coluna de Proventos
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_12meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_24meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_36meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_48meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_60meses", 0)

    ## Adicionar os proventos
    df_Tratar_por_Acao.loc[:, "Prov_12meses"] = df_Tratar_por_Acao.apply(lambda x: x["DY_12m"]*x["Preco_fechamento"], axis=1)
    df_Tratar_por_Acao.loc[:, "Prov_24meses"] = df_Tratar_por_Acao.apply(lambda x: x["DY_24m"]*x["Preco_fechamento"], axis=1)
    df_Tratar_por_Acao.loc[:, "Prov_36meses"] = df_Tratar_por_Acao.apply(lambda x: x["DY_36m"]*x["Preco_fechamento"], axis=1)
    df_Tratar_por_Acao.loc[:, "Prov_48meses"] = df_Tratar_por_Acao.apply(lambda x: x["DY_48m"]*x["Preco_fechamento"], axis=1)
    df_Tratar_por_Acao.loc[:, "Prov_60meses"] = df_Tratar_por_Acao.apply(lambda x: x["DY_60m"]*x["Preco_fechamento"], axis=1)

    # Computar os desdobramentos, grupamentos e Bonificações
    if Existe_eventos:
        ## Datas do Eventos com a condição de Classe de Ação
        datas_eventos = df_eventos.loc[:].index


        ## Loop para percorrer as datas dos eventos e alterar o número equivalente de ações
        for data in datas_eventos:
            
            ## Fator do evento
            condicao_even = (df_eventos.index == data)
            fator_evento = df_eventos.loc[condicao_even,"Fator"].prod()

            ## Condição para recalcular o número de ações equivalentes, no df_Tratar_por_Acao
            ## Lembrar que "data" é a "data-com" do evento, portanto está incluída
            condicao = df_Tratar_por_Acao.index <= data
            ## Recalcular o número de ações equivalentes
            df_Tratar_por_Acao.loc[condicao,"Num_acoes_equivalentes"] = df_Tratar_por_Acao.loc[condicao,"Num_acoes_equivalentes"]/fator_evento

            ## Recalcular os proventos
            df_Tratar_por_Acao.loc[condicao,"Prov_12meses"] = df_Tratar_por_Acao.loc[condicao,"Prov_12meses"]*fator_evento
            df_Tratar_por_Acao.loc[condicao,"Prov_24meses"] = df_Tratar_por_Acao.loc[condicao,"Prov_24meses"]*fator_evento
            df_Tratar_por_Acao.loc[condicao,"Prov_36meses"] = df_Tratar_por_Acao.loc[condicao,"Prov_36meses"]*fator_evento
            df_Tratar_por_Acao.loc[condicao,"Prov_48meses"] = df_Tratar_por_Acao.loc[condicao,"Prov_48meses"]*fator_evento
            df_Tratar_por_Acao.loc[condicao,"Prov_60meses"] = df_Tratar_por_Acao.loc[condicao,"Prov_60meses"]*fator_evento




    ## Adicionar o Ticker
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Fonte", "Comdinheiro")


    ## Normalizar os dados pelo número de ações equivalentes
    colunas_divididas = ['PL', 'RL', 'EBITDA', 'D&A', 'EBIT', 'LL',
                        'LL_controlador', 'LL_nao_controlador', 
                        'Div_Bruta', 'Div_liq', 'Div_Arrendamento', 'FCO', 'FCI', 'FCF']
    for col in colunas_divididas:
        df_Tratar_por_Acao.loc[:, col] = df_Tratar_por_Acao.apply(lambda row: row[col]/row['Num_acoes_equivalentes'], axis=1)

    return df_Tratar_por_Acao

# %%
def ajuste_cotacoes(df_cot, df_Tratar_por_Acao, df_eventos, Existe_eventos, Ticker):
    ## Tratar o dataframe das cotações com os eventos
    df_cot_tratado = df_cot.loc[:,["Fech_Historico","Fech_Ajustado"]].copy()
    df_cot_tratado.columns = ["Fechamento_Equivalente","Fech_Ajustado"]
    df_cot_tratado.insert(0,"Num_acoes_equivalentes",0)

    ## Filtrar as datas do df_Tratar_por_Acao menores que a data do df_cot_tratado
    datas_fund = df_Tratar_por_Acao.index
    datas_cot = df_cot_tratado.index
    ## Loop para percorrer as datas dos eventos e alterar o número equivalente de ações
    ## Considerando a Classe da ação. Lembrar que o Num_acoes_equivalentes calculado anteriormente
    ## só foi calculado para cada trimestre, aqui considera o número de ações equivalentes para cada dia
    for data in datas_cot:
        condicao = datas_fund <= data
        # Se todos forem falso pode deletar a linha
        if condicao.sum() > 0:
            Num_acoes = df_Tratar_por_Acao.loc[condicao,"Num_acoes"][0]
            Fator_equivalencia_acoes = df_Tratar_por_Acao.loc[condicao,"Fator_equivalencia_acoes"][0]
            df_cot_tratado.loc[data,"Num_acoes_equivalentes"] = Num_acoes / Fator_equivalencia_acoes
            #print(data, Num_acoes, Fator_equivalencia_acoes)
        else:
            df_cot_tratado.drop(data, inplace=True)
            Num_acoes = 0
            Fator_equivalencia_acoes = 0


    ## Normalizar os dados pelo número de ações equivalentes, considerando os eventos

    if Existe_eventos:
        datas_eventos = df_eventos.index
        for data in datas_eventos:
            
            ## Fator do evento
            condicao_evento = (df_eventos.index == data)
            fator_evento = df_eventos.loc[condicao_evento,"Fator"].prod()

            ## Condição para recalcular o Preço de Fechamento Histórico, no df_cot_tratado
            condicao = df_cot_tratado.index <= data
            ## Recalcular o preço Equivalente
            df_cot_tratado.loc[condicao,"Fechamento_Equivalente"] = df_cot_tratado.loc[condicao,"Fechamento_Equivalente"]*fator_evento

            ## Recalcular o número de ações equivalentes
            df_cot_tratado.loc[condicao,"Num_acoes_equivalentes"] = df_cot_tratado.loc[condicao,"Num_acoes_equivalentes"]/fator_evento
        
    ## Adicionar o Ticker
    df_cot_tratado.insert(0, "Ticker", Ticker)
    # Adicionar o Coluna de Market Value
    df_cot_tratado.insert(len(df_cot_tratado.columns), "Market_value", 0)
    df_cot_tratado.loc[:,"Market_value"] = df_cot_tratado.apply(lambda row: row["Fechamento_Equivalente"]*row["Num_acoes_equivalentes"], axis=1)

    # Fonte
    df_cot_tratado.insert(len(df_cot_tratado.columns), "Fonte", "Comdinheiro")
    
    return df_cot_tratado


# %%
def ajuste_prov(df_prov, df_eventos, Existe_prov, Existe_eventos):
    

    ## Considerar os eventos de desdobramentos, grupamentos e bonificações
    if Existe_prov:
        ## Tratar o dataframe de proventos com os eventos
        df_prov_tratado = df_prov.copy()
        df_prov_tratado.insert(0,"Provento_Efeitivo",0)

        ## Descontar imposto do JCP
        df_prov_tratado.loc[:, "Provento_Efeitivo"] = \
            df_prov_tratado.apply(lambda row: row["Valor_do_Provento"]*0.85 if row["Tipo_do_Provento"]=="JCP" else row["Valor_do_Provento"], axis=1)
        
        if Existe_eventos:
            datas_eventos = df_eventos.index
            for data in datas_eventos:
                
                ## Fator do evento
                condicao = (df_eventos.index == data)
                fator_evento = df_eventos.loc[condicao,"Fator"].prod()

                ## Condição para recalcular o Preço de Fechamento Histórico, no df_cot_tratado
                condicao = df_prov_tratado.index <= data
                ## Recalcular o número de ações equivalentes
                df_prov_tratado.loc[condicao,"Provento_Efeitivo"] = df_prov_tratado.loc[condicao,"Provento_Efeitivo"]*fator_evento

        return df_prov_tratado


# %% [markdown]
# ## Múltiplos

# %%
def multiplos_diarios(df_Tratar_por_Acao, df_cot_tratado, Ticker):
    # Coletando as datas de balanço e cotação
    data_cotacao = df_cot_tratado.index
    data_balanco = df_Tratar_por_Acao.index[0:-4]
    primeiro_balanco = data_balanco[-1]
    data_multiplos_diarios = data_cotacao[data_cotacao >= primeiro_balanco]

    # Criando um DataFrame de múltiplos com as datas de balanço e cotação, Trimetre e Anual
    df_multiplos_diarios_tri = pd.DataFrame(index=data_multiplos_diarios,
    columns=["Num_acoes_equivalentes","Fechamento_Equivalente","Fech_Ajustado", 
             "Market_value", "EV", "EV_arrend",
            "PVPA","PSR","EV_EBITDA","EV_EBITDA_Arr","P_EBIT","PE", "PE_C",
            "FCO","FCI","FCF",
            "ROE", "Margem_liquida","Margem_EBITDA",
            "DIV_Bruta_PL","DIV_liq_EBITDA","DIV_Arrendamento_EBITDA",
            'DY_12m', 'DY_24m', 'DY_36m', 'DY_48m', 'DY_60m', "DY_medio" ,"Fonte"])


    ## Preenchendo os dados de cotação
    df_multiplos_diarios_tri.insert(0, "Ticker", Ticker)
    df_multiplos_diarios_tri.loc[:, "Fonte"] = "Comdinheiro"

    # Preenchendo alguns valores
    lista_colunas = ["Num_acoes_equivalentes","Fechamento_Equivalente","Fech_Ajustado", "Market_value"]
    df_multiplos_diarios_tri.loc[:, lista_colunas]\
        = df_cot_tratado.loc[:, lista_colunas]

    ## Criano o DF de múltiplos diários com os dados fundamentalistas anuais
    df_multiplos_diarios_anual = df_multiplos_diarios_tri.copy()

    # Preenchendo os valores Absolutos
    for data in data_multiplos_diarios:
        condicao = df_Tratar_por_Acao.loc[:,"Data_balanco"] <= data
        #print(data, condicao)
        if condicao.sum()>0:
            Preco_equivalente = df_multiplos_diarios_tri.loc[data,"Fechamento_Equivalente"]
            ## EV = Market_value + Div_liq; Atenção que não foi considerado a dívida de arrendamento
            EV = Preco_equivalente + df_Tratar_por_Acao.loc[condicao,"Div_liq"][0]
            EV_Arrend = EV + df_Tratar_por_Acao.loc[condicao,"Div_Arrendamento"][0]

            ## Atribuindo os valores de EV
            df_multiplos_diarios_tri.loc[data,"EV"] = EV
            df_multiplos_diarios_anual.loc[data,"EV"] = EV
            df_multiplos_diarios_tri.loc[data,"EV_arrend"] = EV_Arrend
            df_multiplos_diarios_anual.loc[data,"EV_arrend"] = EV_Arrend

            ## PVPA -> Preço / PL
            PL = df_Tratar_por_Acao.loc[condicao,"PL"][0]
            PVPA = Preco_equivalente / PL
            df_multiplos_diarios_tri.loc[data,"PVPA"] = PVPA
            df_multiplos_diarios_anual.loc[data,"PVPA"] = PVPA


            ## Dados Fundamentalistas de 1 ano
            Receita_12meses = df_Tratar_por_Acao.loc[condicao,"RL"][:4].sum()
            EBITDA_12meses = df_Tratar_por_Acao.loc[condicao,"EBITDA"][:4].sum()
            EBIT_12meses = df_Tratar_por_Acao.loc[condicao,"EBIT"][:4].sum()
            LL_12meses = df_Tratar_por_Acao.loc[condicao,"LL"][:4].sum()
            LL_C_12meses = df_Tratar_por_Acao.loc[condicao,"LL_controlador"][:4].sum()
            Div_liq_12meses = df_Tratar_por_Acao.loc[condicao,"Div_liq"][:1].sum()
            Div_bruta_12meses = df_Tratar_por_Acao.loc[condicao,"Div_Bruta"][:1].sum()
            Div_liq_Arr_12meses = Div_liq_12meses + df_Tratar_por_Acao.loc[condicao,"Div_Arrendamento"][:1].sum()
            FCO_12meses = df_Tratar_por_Acao.loc[condicao,"FCO"][:4].sum()
            FCI_12meses = df_Tratar_por_Acao.loc[condicao,"FCI"][:4].sum()
            FCF_12meses = df_Tratar_por_Acao.loc[condicao,"FCF"][:4].sum()


            ## Dados Fundamentalistas de 3 meses
            Receita_3meses = df_Tratar_por_Acao.loc[condicao,"RL"][:1].sum()*4
            EBITDA_3meses = df_Tratar_por_Acao.loc[condicao,"EBITDA"][:1].sum()*4
            EBIT_3meses = df_Tratar_por_Acao.loc[condicao,"EBIT"][:1].sum()*4
            LL_3meses = df_Tratar_por_Acao.loc[condicao,"LL"][:1].sum()*4
            LL_C_3meses = df_Tratar_por_Acao.loc[condicao,"LL_controlador"][:1].sum()*4
            Div_liq_3meses = Div_liq_12meses
            Div_bruta_3meses = Div_bruta_12meses
            Div_liq_Arr_3meses = Div_liq_Arr_12meses
            FCO_3meses = df_Tratar_por_Acao.loc[condicao,"FCO"][:1].sum()*4
            FCI_3meses = df_Tratar_por_Acao.loc[condicao,"FCI"][:1].sum()*4
            FCF_3meses = df_Tratar_por_Acao.loc[condicao,"FCF"][:1].sum()*4

            ## Registrando os dados de 1 ano
            df_multiplos_diarios_anual.loc[data,"PSR"] = Preco_equivalente / Receita_12meses if Receita_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"EV_EBITDA"] = EV / EBITDA_12meses if EBITDA_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"EV_EBITDA_Arr"] = EV_Arrend / EBITDA_12meses if EBITDA_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"P_EBIT"] = Preco_equivalente / EBIT_12meses if EBIT_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"PE"] = Preco_equivalente / LL_12meses if LL_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"PE_C"] = Preco_equivalente / LL_C_12meses if LL_C_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"FCO"] = Preco_equivalente / FCO_12meses if FCO_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"FCI"] = Preco_equivalente / FCI_12meses if FCI_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"FCF"] = Preco_equivalente / FCF_12meses if FCF_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"ROE"] = LL_12meses / PL if PL != 0 else 0
            df_multiplos_diarios_anual.loc[data,"Margem_liquida"] = LL_12meses / Receita_12meses if Receita_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"Margem_EBITDA"] = EBITDA_12meses / Receita_12meses if Receita_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"DIV_Bruta_PL"] = Div_bruta_12meses / PL if PL != 0 else 0
            df_multiplos_diarios_anual.loc[data,"DIV_liq_EBITDA"] = Div_liq_12meses / EBITDA_12meses if EBITDA_12meses != 0 else 0
            df_multiplos_diarios_anual.loc[data,"DIV_Arrendamento_EBITDA"] = Div_liq_Arr_3meses / EBITDA_12meses if EBITDA_12meses != 0 else 0


            ## Registrando os dados de 3 meses
            df_multiplos_diarios_tri.loc[data,"PSR"] = Preco_equivalente / Receita_3meses if Receita_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"EV_EBITDA"] = EV / EBITDA_3meses if EBITDA_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"EV_EBITDA_Arr"] = EV_Arrend / EBITDA_3meses if EBITDA_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"P_EBIT"] = Preco_equivalente / EBIT_3meses if EBIT_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"PE"] = Preco_equivalente / LL_3meses if LL_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"PE_C"] = Preco_equivalente / LL_C_3meses if LL_C_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"FCO"] = Preco_equivalente / FCO_3meses if FCO_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"FCI"] = Preco_equivalente / FCI_3meses if FCI_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"FCF"] = Preco_equivalente / FCF_3meses if FCF_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"ROE"] = LL_3meses / PL if PL != 0 else 0
            df_multiplos_diarios_tri.loc[data,"Margem_liquida"] = LL_3meses / Receita_3meses if Receita_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"Margem_EBITDA"] = EBITDA_3meses / Receita_3meses if Receita_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"DIV_Bruta_PL"] = Div_bruta_3meses / PL if PL != 0 else 0
            df_multiplos_diarios_tri.loc[data,"DIV_liq_EBITDA"] = Div_liq_3meses / EBITDA_3meses if EBITDA_3meses != 0 else 0
            df_multiplos_diarios_tri.loc[data,"DIV_Arrendamento_EBITDA"] = Div_liq_Arr_3meses / EBITDA_3meses if EBITDA_3meses != 0 else 0
            


            ## Adicionar os proventos
            df_multiplos_diarios_tri.loc[data, "DY_12m":"DY_60m"] = 0
            # Proventos
            Prov_1ano = df_Tratar_por_Acao.loc[condicao, "Prov_12meses"][0]
            Prov_2ano = df_Tratar_por_Acao.loc[condicao, "Prov_24meses"][0]/2
            Prov_3ano = df_Tratar_por_Acao.loc[condicao, "Prov_36meses"][0]/3
            Prov_4ano = df_Tratar_por_Acao.loc[condicao, "Prov_48meses"][0]/4
            Prov_5ano = df_Tratar_por_Acao.loc[condicao, "Prov_60meses"][0]/5

            ## Salvando o DY médio

            df_multiplos_diarios_anual.loc[data, "DY_12m"] = Prov_1ano/ Preco_equivalente
            df_multiplos_diarios_anual.loc[data, "DY_24m"] = Prov_2ano/ (Preco_equivalente)
            df_multiplos_diarios_anual.loc[data, "DY_36m"] = Prov_3ano/ (Preco_equivalente)
            df_multiplos_diarios_anual.loc[data, "DY_48m"] = Prov_4ano/ (Preco_equivalente)
            df_multiplos_diarios_anual.loc[data, "DY_60m"] = Prov_5ano/ (Preco_equivalente)

            df_multiplos_diarios_anual.loc[data, "DY_medio"] = (Prov_2ano + Prov_3ano + Prov_4ano + Prov_5ano)/ (Preco_equivalente*4) ## Descartar o último ano para evitar distorções


    return df_multiplos_diarios_tri, df_multiplos_diarios_anual




# %% [markdown]
# ## Funções para estimar o crescimento

# %%
def CalculoCAGRnAnos(serie):

    copSerie = serie.copy()
    menor_valor = copSerie.min()
    maior_valor = copSerie.max()
    copSerie.sort_index(ascending=True, inplace=True)

    if maior_valor < 0:
        print("Maior Valor menor que zero")
        return -1.01
    elif menor_valor < 0:
        copSerie = copSerie + (maior_valor- menor_valor)/2

    copSerie = copSerie.apply(lambda x: np.log(x))
    try:
        expoente = np.polyfit(range(len(copSerie)), copSerie.values, 1)[0] # Highest power first
    except:
        expoente = 1
        print("Erro no calculo do CAGR")
        print(serie)
    CAGR = (np.exp(expoente))**4 -1 
    return CAGR

# %%
def Func_CAGR(df_Tratar_por_Acao, Ticker):
    df_CAGR = pd.DataFrame(columns=["Ticker","Fundamento","CAGR_1","CAGR_2","CAGR_4","CAGR_8"])
    Lista_Fundamentos = ["RL", "EBITDA", "LL", "Proventos"]
    Lista_anos_CAGR = [1,2,4,8]

    for fundamento in Lista_Fundamentos:
        dados_fundamentalistas = df_Tratar_por_Acao.loc[:,fundamento]
        dados_fundamentalistas = dados_fundamentalistas.copy()
        df_CAGR_temp = pd.DataFrame(index=df_Tratar_por_Acao.index, columns=["Ticker","Fundamento","CAGR_1","CAGR_2","CAGR_4","CAGR_8"])
        df_CAGR_temp.loc[:, "Ticker"] = Ticker
        df_CAGR_temp.loc[:, "Fundamento"] = fundamento
        for qtd_anos in Lista_anos_CAGR:
            for data in df_Tratar_por_Acao.index:
                ano = data.year
                mes = data.month
                dia = data.day
                ## Data de início do corte
                data_inicio = datetime(ano - qtd_anos, mes, dia-10)

                condicao = (dados_fundamentalistas.index >= data_inicio) & \
                (dados_fundamentalistas.index <= data)

                ## Cálculo do CAGR
                serie = dados_fundamentalistas.loc[condicao]
                if len(serie) >= qtd_anos*4:
                    CAGR = CalculoCAGRnAnos(serie)
                else:
                    CAGR = 0

                ## Salvando os dados no df_CAGR
                df_CAGR_temp.loc[data, f"CAGR_{qtd_anos}"] = CAGR

        ##  Salvando os dados no df_CAGR
        df_CAGR = pd.concat([df_CAGR, df_CAGR_temp], axis=0)

    ## Consertar os dados do CAGR do Provento
    df_CAGR.loc[df_CAGR["Fundamento"]=="Proventos", "CAGR_1":"CAGR_8"] = df_CAGR.loc[df_CAGR["Fundamento"]=="Proventos", "CAGR_1":"CAGR_8"].apply(lambda x: (1+x)**0.25-1)

    ## Se CAGR for maior do que 50% considerar como 50%
    CAGR_Maximo = 2
    df_CAGR.loc[df_CAGR["CAGR_1"] > CAGR_Maximo, "CAGR_1"] = CAGR_Maximo
    df_CAGR.loc[df_CAGR["CAGR_2"] > CAGR_Maximo, "CAGR_2"] = CAGR_Maximo
    df_CAGR.loc[df_CAGR["CAGR_4"] > CAGR_Maximo, "CAGR_4"] = CAGR_Maximo
    df_CAGR.loc[df_CAGR["CAGR_8"] > CAGR_Maximo, "CAGR_8"] = CAGR_Maximo
    

    ## Calculando a média
    df_CAGR.loc[:, "CAGR_medio"] = df_CAGR.loc[:, "CAGR_2":"CAGR_8"].mean(axis=1)  ## Descartar o último ano para evitar distorções
    return df_CAGR

# %% [markdown]
# ## Função para tratar dados diários

# %%
def Tratar_dados_diarios(Ticker):
    ## Endereço dos arquivos
    arquivo_Fund, arquivo_Cot, arquivo_Prov, arquivo_Eventos, \
        arquivo_Subscricao = endereco_arquivos(Ticker)
    
    print(f"Entrou em {Ticker}")
    ## Verificando a classe da ação
    Classe_acao = Func_Classe_acao(Ticker)

    ## Tratar dados Fundamentalistas
    df_fund = tratar_Fund(arquivo_Fund)

    ## Tratar dados de cotações
    df_cot = tratar_cot(arquivo_Cot)

    ## Tratar dados de proventos
    df_prov, Existe_prov = tratar_prov(arquivo_Prov, Classe_acao)

    ## Tratar dados de eventos
    df_eventos, Existe_eventos = tratar_even(arquivo_Eventos, Classe_acao)

    ## Tratar dados de subscrições
    df_subscricao, Existe_subscricao = tratar_subs(arquivo_Subscricao)

    ## Normalizar Dados Fundamentalistas e considerar os eventos
    df_Tratar_por_Acao = normalizar_dados_fund(df_fund, df_eventos, Existe_eventos, Ticker)

    ## Ajustar as cotações
    df_cot_tratado = ajuste_cotacoes(df_cot, df_Tratar_por_Acao, df_eventos, Existe_eventos, Ticker)


    ## Ajustar os proventos
    df_prov_tratado = ajuste_prov(df_prov, df_eventos, Existe_prov, Existe_eventos)

    ## Multiplos diários
    df_multiplos_diarios_tri, df_multiplos_diarios_anual = multiplos_diarios(df_Tratar_por_Acao, df_cot_tratado, Ticker)

    ## CAGR
    df_CAGR = Func_CAGR(df_Tratar_por_Acao, Ticker)

    ### Salvar os arquivos em parquet
    # Salvar Fundamentos
    nome_arquivo = f"Dados_normalizados_acao_{Ticker}.parquet"
    arquivo_por_acao = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Tratados",nome_arquivo)
    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    df_Tratar_por_Acao.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_Tratar_por_Acao.to_parquet(arquivo_por_acao, engine='fastparquet')

    ## Salvar Múltiplos diários
    nome_arquivo = f"Multiplos_diarios_{Ticker}.parquet"
    arquivo_multiplos = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Tratados",nome_arquivo)
    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    df_multiplos_diarios_anual.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_multiplos_diarios_anual.to_parquet(arquivo_multiplos, engine='fastparquet')

    ## Salvar o CAGR
    nome_arquivo = f"CAGR_{Ticker}.parquet"
    arquivo_CAGR = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Tratados",nome_arquivo)
    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    df_CAGR.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_CAGR.to_parquet(arquivo_CAGR, engine='fastparquet')

    
    return df_Tratar_por_Acao, df_multiplos_diarios_anual, df_CAGR

# %% [markdown]
# ### Lendo os arquivos dos dados Brutos

# %%
## Ler os ativos que serão buscados
arquivo_busca = os.path.join("..","..","..","dataset", "BR", "ACOES", "Dados_Brutos","Lista_Ativos_Busca.parquet")
Lista_ativos_busca = pd.read_parquet(arquivo_busca)
Lista_ativos_busca = Lista_ativos_busca["Ticker"].to_list()
Lista_ativos_busca.sort()
Lista_ativos_nao_deu_certo = []

for Ticker in Lista_ativos_busca:
    try:
        df_Tratar_por_Acao, df_multiplos_diarios_anual, df_CAGR = Tratar_dados_diarios(Ticker)
        print(f"Deu certo {Ticker}")
    except:
        print(f"Não deu certo {Ticker}")
        Lista_ativos_nao_deu_certo.append(Ticker)


