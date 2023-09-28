
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
import os


# Funções

def Regressao_linear(df,x_label,y_label,prev_n_steps_meses,ticker, tipo="linear"):
    """
    Realiza uma regressão linear de um múltiplo em relação à expectativa da SELIC.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    x_label (str): Nome da coluna que contém a expectativa de SELIC no eixo x.
    y_label (str): Nome da coluna que contém o múltiplo no eixo y.
    prev_n_steps_meses (float): Previsão da expectativa da SELIC para 'n' meses no futuro.
    ticker (str): Ticker da ação.
    tipo (str): Tipo de regressão ('linear' ou 'log'). Padrão é 'linear'.

    Retorna:
    fig_regressao (Figure): Gráfico da regressão linear.
    fig_residuos (Figure): Gráfico de resíduos da regressão.
    r_squared (float): Coeficiente de determinação (R²) da regressão.
    y_ultimo_prev (float): Valor previsto do múltiplo para a expectativa de SELIC fornecida.
    summary (str): Resumo estatístico do modelo de regressão.

    A função realiza uma regressão linear ou logarítmica dos valores do múltiplo em relação à expectativa da SELIC.
    Em seguida, calcula o valor previsto do múltiplo para a expectativa de SELIC fornecida e cria gráficos da regressão
    e dos resíduos. Retorna o R², o valor previsto e um resumo estatístico do modelo.

    Exemplo de uso:
    fig_regressao, fig_residuos, r_squared, y_ultimo_prev, summary = Regressao_linear(df, 'Expectativa_SELIC', 'Multiplo', 12, 'PETR4', 'linear')
    """

    
    x = df[[x_label]]

    if tipo=="linear":
        y = df[y_label]
    elif tipo=="log":
        logy = np.log(df[y_label])
        y = logy
    
    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    try:
        # modelo 
        model = sm.OLS(y, x).fit()
    except:
        raise
    parametros = model.params
    r_squared = model.rsquared
    intercept = parametros["const"]
    coef_x = parametros[x_label] 
    # Calcule os resíduos
    residuos = model.resid
    # Calcule o desvio padrão dos resíduos
    desvio_padrao_residuos = np.std(residuos)

    # Obtenha o resumo do modelo
    summary = model.summary()

    # Fazendo uma reta com os parâmetros encontrados
    x_RANGE = np.linspace(1.5,17,100)
    y = x_RANGE*coef_x+ intercept
    y_ultimo_prev = coef_x*prev_n_steps_meses+ intercept

    ## Desvio padrão do plot
    um_desvio_sup = y + desvio_padrao_residuos
    um_desvio_inf = y - desvio_padrao_residuos



    if tipo=="log":
        y_ultimo_prev = np.exp(y_ultimo_prev)
        um_desvio_sup = np.exp(um_desvio_sup)
        um_desvio_inf = np.exp(um_desvio_inf)
        y = np.exp(y)

    y_ultimo_prev = round(y_ultimo_prev,3)

    # Gráfico da regressão linear

    fig_regressao, ax = plt.subplots(figsize=(15,8))

    ax.scatter(df[x_label],df[y_label])
    ax.scatter(df[x_label][-1],df[y_label][-1],color='red', label="Último Ponto")

    
    # Adicione uma linha vertical em x = 3
    ultm_prev_SELIC = prev_n_steps_meses
    ax.axvline(x=ultm_prev_SELIC, color='green', linestyle='--', label=f'Linha Vertical, {round(ultm_prev_SELIC,2)} %')
    ax.axhline(y=y_ultimo_prev, color='red', linestyle='--', label=f'Linha Horizontal, {y_ultimo_prev}')
    ax.plot(x_RANGE,y,color='black',label=f'Regressão Linear')
    ax.plot(x_RANGE, um_desvio_sup, linestyle='--', color='blue', label='1 Desvio Padrão Superior')
    ax.plot(x_RANGE, um_desvio_inf, linestyle='--', color='blue', label='1 Desvio Padrão Inferior')

    ax.legend()
    plt.xlabel("Expectativa de SELIC")
    plt.ylabel(f"{y_label}")
    plt.title(f"Ticker: {ticker}, {y_label} x Expectativa de SELIC com R² = {round(r_squared,3)}")
    plt.close(fig_regressao)

    ## Plote do resíduos lineares
    # Crie um gráfico de dispersão de resíduos versus valores previstos
    fig_residuos, ax = plt.subplots(figsize=(15,8))
    ax.scatter(model.fittedvalues, residuos)
    ax.set_xlabel("Valores Previstos")
    ax.set_ylabel("Resíduos")
    ax.set_title(f"Ticker: {ticker},Gráfico de Resíduos")

    # Adicione linhas tracejadas em múltiplos desvios padrão (por exemplo, 1 e 2 desvios padrão)
    ax.axhline(y=desvio_padrao_residuos, color='r', linestyle='--', label='1 Desvio Padrão')
    ax.axhline(y=-desvio_padrao_residuos, color='r', linestyle='--')
    ax.axhline(y=2 *desvio_padrao_residuos, color='g', linestyle='--', label='2 Desvios Padrão')
    ax.axhline(y=-2 * desvio_padrao_residuos, color='g', linestyle='--')

    # Adicione uma linha horizontal em y=0 para facilitar a visualização dos resíduos
    ax.axhline(y=0, color='b', linestyle='-')

    # Adicione uma legenda
    plt.legend()
    plt.close(fig_residuos)


    return fig_regressao, fig_residuos, r_squared, y_ultimo_prev, summary


def Func_definir_melhor_regressao(df,x_label,y_label,prev_n_steps_meses, ticker):
    """
    Analisa e compara duas regressões lineares (linear e logarítmica) entre a expectativa da SELIC e um múltiplo desejado.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    x_label (str): Nome da coluna que contém a expectativa da SELIC no eixo x.
    y_label (str): Nome da coluna que contém o múltiplo desejado no eixo y.
    prev_n_steps_meses (float): Previsão da expectativa da SELIC para 'n' meses no futuro.
    ticker (str): Ticker da ação.

    Retorna:
    lista_figuras (list): Lista de figuras geradas (gráficos).
    y_ultimo_prev (float): Valor previsto do múltiplo com base na melhor regressão.
    p_valor (float): P-valor do coeficiente do melhor modelo de regressão.
    r_squared (float): Coeficiente de determinação (R²) do melhor modelo de regressão.
    summary (str): Resumo estatístico do melhor modelo de regressão.

    A função cria gráficos da expectativa da SELIC e do múltiplo desejado, realiza regressões lineares (linear e logarítmica),
    compara os R² das regressões e retorna os resultados da melhor regressão.

    Exemplo de uso:
    lista_figuras, y_ultimo_prev, p_valor, r_squared, summary = Func_definir_melhor_regressao(df, 'Expectativa_SELIC', 'Multiplo', 12, 'PETR4')
    """
    
    ## Lista de figuras
    lista_figuras = []
    
    ## Gráfico do múltiplo
    try:
        fig_multiplo_selic, ax = plt.subplots(figsize=(15,6))
        ax.plot(df.index, df[x_label],label=x_label)
        # Plotar no eixo secundário
        ax2 = ax.twinx()
        ax2.plot(df.index, df[y_label],color="red", label=y_label)
        ax2.axhline(y=df[y_label].mean(), color='r', linestyle='--', label=f'Média do{y_label}')
        # Adicione linhas tracejadas em múltiplos desvios padrão (por exemplo, 1 e 2 desvios padrão)
        ax2.axhline(y=df["PVPA"].mean()+df["PVPA"].std(), color='g', linestyle='-.', label='1 Desvio Padrão')
        ax2.axhline(y=df["PVPA"].mean()-df["PVPA"].std(), color='g', linestyle='-.')
        
        # Ativar a legenda
        ax.legend(loc=2)
        ax2.legend(loc=1)
        # Adicione um título ao gráfico
        ax.set_title(f"{ticker}, análise de {x_label} e {y_label}.")
        # Especifique os valores de x (datas) que você deseja exibir no eixo x (por exemplo, a cada ano)
        valores_xticks = pd.date_range(start=df.index.min(), end=df.index.max(), freq="A")
        # Atribua rótulos formatados para os valores de x
        rotulos_xticks = [data.strftime("%Y") for data in valores_xticks]
         # Configure os valores e rótulos no eixo x
        ax.set_xticks(valores_xticks)
        ax.set_xticklabels(rotulos_xticks, rotation=45)  # A rotação é opcional para melhor legibilidade

        plt.close(fig_multiplo_selic)
        lista_figuras.append(fig_multiplo_selic)
    except:
        raise

    ## (FIM) Gráfico do Múltiplo

     ## Fazendo a regressão Linear
    fig_regressao, fig_residuos, r_squared_lin, y_ultimo_prev_lin, summary_lin = \
        Regressao_linear(df,x_label,y_label,prev_n_steps_meses, ticker,tipo="linear")
    lista_figuras.append(fig_regressao)
    lista_figuras.append(fig_residuos)
    # P-valor do coeficiente Valor
    p_valor_coeficiente_lin = summary_lin.tables[1].data[2][4]
    ## Fazendo a regressão Log
    fig_regressao, fig_residuos, r_squared_log, y_ultimo_prev_log, summary_log = \
        Regressao_linear(df,x_label,y_label,prev_n_steps_meses, ticker, tipo="log")
    lista_figuras.append(fig_regressao)
    lista_figuras.append(fig_residuos)
    # P-valor do coeficiente Valor
    p_valor_coeficiente_log = summary_log.tables[1].data[2][4]


    ## Ver qual regressão teve o melhor R^2
    if r_squared_lin>=r_squared_log:
        y_ultimo_prev = y_ultimo_prev_lin
        p_valor = p_valor_coeficiente_lin
        r_squared = r_squared_lin
        summary = summary_lin
       
    else:
        y_ultimo_prev = y_ultimo_prev_log
        p_valor = p_valor_coeficiente_log
        r_squared = r_squared_log
        summary = summary_log

    return lista_figuras, y_ultimo_prev, p_valor, r_squared, summary

def weighted_mean_and_std(series):
    """
    Calcula a média ponderada e o desvio padrão ponderado de uma série temporal com base no tempo.

    Parâmetros:
    series (pd.Series): Série temporal com índice contendo datas.

    Retorna:
    weighted_mean (float): Média ponderada da série.
    weighted_std (float): Desvio padrão ponderado da série.

    A função calcula a média ponderada e o desvio padrão ponderado de uma série temporal, onde os valores mais recentes
    têm maior peso e os pesos diminuem em 0.9 a cada 365 dias.

    Exemplo de uso:
    weighted_mean, weighted_std = weighted_mean_and_std(series)
    """
    # Crie uma série de datas de referência
    datas = series.index

    ## Data da simulação alterada
    data_simulacao = series.index.max()
    data_simulacao

    # Calcule as diferenças em dias entre cada data da simulação e a cada data
    time_diff = (data_simulacao - datas).days
    time_diff = np.array(time_diff)

    # Calcule os pesos com base na redução em 0.9 a cada 365 dias
    weights = 0.9 ** (time_diff / 365)

    # Calcule a média ponderada e o desvio padrão ponderado
    weighted_mean = np.average(series, weights=weights)
    weighted_var = np.average((series - weighted_mean) ** 2, weights=weights)
    weighted_std = np.sqrt(weighted_var)

    return weighted_mean, weighted_std

def Dados_iniciais(df_acao, df_multiplos, df_CAGR, Fundamento, multiplo):
    """
    Coleta dados iniciais necessários para o cálculo de outras funções.

    Parâmetros:
    df_acao (pd.DataFrame): DataFrame contendo dados fundamentalistas da ação normalizados pelo número de ações ex-tesouraria.
    df_multiplos (pd.DataFrame): DataFrame contendo os múltiplos da empresa (dados diários).
    df_CAGR (pd.DataFrame): DataFrame contendo a taxa de crescimento de dados fundamentalistas.
    Fundamento (str): Fundamento de crescimento a ser analisado (por exemplo, 'RL' para receita líquida).
    multiplo (str): Múltiplo a ser avaliado (por exemplo, 'PE' para price earnings, 'EV_EBITDA' para EV/EBITDA).

    Retorna:
    dados_iniciais (list): Lista de dados iniciais coletados para uso em outras funções.

    A função coleta informações como setor, ROE, DY médio, ticker, CAGR médio, múltiplo atual, múltiplo médio,
    desvio do múltiplo médio e expansão do múltiplo. Esses dados são usados em outras funções para análise.

    Exemplo de uso:
    dados_iniciais = Dados_iniciais(df_acao, df_multiplos, df_CAGR, 'RL', 'PE')
    """
    ## Data da simulação alterada
    data_simulacao = df_multiplos.index.max()

    ## Coletando dados iniciais
    Setor = df_acao.loc[:,"Setor_Comdinheiro"][0]
    ROE = df_multiplos.loc[data_simulacao,"ROE"]
    DY_medio = df_multiplos.loc[data_simulacao,"DY_medio"]
    ticker = df_acao["Ticker"][0]    
    # Crescimento do lucro
    condicao1 = df_CAGR.index == df_CAGR.index.max()
    condicao2 = df_CAGR["Fundamento"] == Fundamento
    condicao = condicao1 & condicao2
    CAGR_Medio = df_CAGR.loc[condicao,"CAGR_medio"].values[0]

    # Multiplo atual
    Multiplo_atual = df_multiplos.loc[df_multiplos.index.max(),multiplo]

    # Multiplo médio
    
    datas_recentes = df_multiplos.index.sort_values(ascending=False)[:8*252] # 8 anos
    serie = df_multiplos.loc[datas_recentes,multiplo]
    Multiplo_medio, STD_multiplo = weighted_mean_and_std(serie)

    
    # Quanto que desviou do múltiplo médio
    Desvio_multiplo = (Multiplo_atual-Multiplo_medio)/STD_multiplo

    # Expansão do múltiplo
    ## Penalizar se o desvio for maior que 0.5
    ## Só vai expandir se o desvio for menor que -1
    if Desvio_multiplo>0.5:
        Expansao_Multiplo = (Multiplo_medio + 0.5*STD_multiplo)/Multiplo_atual
    elif Desvio_multiplo<-1:
        Expansao_Multiplo = (Multiplo_medio - 1*STD_multiplo)/Multiplo_atual
    else:
        Expansao_Multiplo = 1

        


    return [data_simulacao, Setor, ROE, DY_medio, ticker, CAGR_Medio, Multiplo_atual, Multiplo_medio, Desvio_multiplo, Expansao_Multiplo]

## Estratégia baseada em VPA
def ret_VPA(df_acao, df_multiplos, df_CAGR, df_Expectativa_Selic_mensal, prev_n_steps_meses):
    """
    Calcula o retorno esperado de ativos com base na relação entre o VPA (Valor Patrimonial por Ação) de uma ação e a expectativa da taxa SELIC.

    Parâmetros:
    df_acao (pd.DataFrame): DataFrame contendo dados fundamentalistas da ação normalizados pelo número de ações ex-tesouraria.
    df_multiplos (pd.DataFrame): DataFrame contendo os múltiplos da empresa (dados diários).
    df_CAGR (pd.DataFrame): DataFrame contendo a taxa de crescimento de dados fundamentalistas.
    df_Expectativa_Selic_mensal (pd.DataFrame): DataFrame contendo as expectativas mensais da taxa SELIC.
    prev_n_steps_meses (float): Valor previsto para a próxima expectativa da taxa SELIC.

    Retorna:
    lista_return (list): Lista de informações, incluindo setor, retorno anual esperado, ROE, dívida bruta/PL, DY médio, CAGR médio, expansão do múltiplo, múltiplo atual, múltiplo médio e desvio do múltiplo.
    lista_figuras (list): Lista de figuras geradas durante a análise.

    A função calcula o retorno esperado dos ativos com base na relação entre o VPA de uma ação e a expectativa da taxa SELIC, considerando penalizações para setores de alta de juros, empresas com muita dívida e r_squared alto.

    Exemplo de uso:
    dados_return, figuras = ret_VPA(df_acao, df_multiplos, df_CAGR, df_Expectativa_Selic_mensal, prev_n_steps_meses)
    """
    ## Coletando dados iniciais
    [data_simulacao, Setor, ROE, DY_medio, ticker, CAGR_Medio, Multiplo_atual, 
     Multiplo_medio, Desvio_multiplo, Expansao_Multiplo] = \
        Dados_iniciais(df_acao, df_multiplos, df_CAGR, "LL", "PVPA")

    

    ## Compilando os dados para fazer a regressão
    df_regressao = df_multiplos[["PVPA"]].copy()
    ## Resample mensal
    df_regressao = df_regressao.resample("M").mean()
    ## Adicionando a expectativa de SELIC
    df_regressao = pd.concat([df_regressao, df_Expectativa_Selic_mensal], axis=1, join="inner")

    ## Filtrar os últimos 8 anos
    n = 12*8
    df_regressao = df_regressao.iloc[-n:,:] # últimos 8 anos
    # Obtendo os dados da Regressão
    lista_figuras, Expansao_multiplo, p_valor, r_squared, summary = \
        Func_definir_melhor_regressao(df_regressao,"Valor","PVPA",prev_n_steps_meses, ticker)
   


    ## Lucro Futuro = Lucro Passado
    ultm_data_balanco = df_acao.index.max()
    LL_ANO = df_acao.loc[ultm_data_balanco, "LL"]

    # Use esses índices para localizar os valores da coluna "PL"
    PL_dia = df_acao.loc[ultm_data_balanco,"PL"]

    ## Última cotação
    Preco_atual = df_multiplos.loc[data_simulacao,"Fechamento_Equivalente"]

    ## PL Futuro = PL Atual + Lucro Futuro
    PL_Futuro = PL_dia + LL_ANO

    ## Previsão de preço
    Preco_previsao = PL_Futuro*Expansao_multiplo**2  # Expansão do múltiplo ao quadrado, portanto, demora 6 meses para se concretizar 

    # Retorno é o preço previsto dividido pelo preço atual
    Retorno = Preco_previsao/Preco_atual
    # Retorno esperado anual
    Retorno_anual_esperado = (Retorno - 1 )

    ## Agora é necessário realizar algumas penalizações, são elas:
    ## penalizacao_hike -> Evitar investimentos neste setor e alta de juros
    ## penalizacao_divida -> Evitar investimentos em empresas com muita dívida
    ## penalizacao_r_squared -> Evitar investimentos em empresas com r_squared muito alto
    
    ## Multiplicador para evitar investir em empresas do setor imobiliário com aumento da SELIC
    ## Desta forma, quando a razão entre Previsto e Atual for maior que 1.03, o beta será 0;
    ## Portanto em momento de hikes deve-se evitar investir em empresas do setor imobiliário
    def penalizacao_hike(razao):
        if razao<=1:
            beta = 1
        elif razao>=1.03:
            beta = 0
        else:
            beta = 1-((razao-1)/(1.03-1))
        return beta
    ## Fim da função beta

    ## Multiplicador para evitar investir em empresas com muita dívida
    ## Se for menor que 0.5, o beta é 1
    ## Se for maior que 1, o beta é 0
    ## Se for entre 0.5 e 1, o beta é 1-((razao-0.5)/(1-0.5))
    def penalizacao_divida(Div_Bruta_por_PL):
        if Div_Bruta_por_PL<=0.5:
            beta = 1
        elif Div_Bruta_por_PL>=1:
            beta = 0
        else:
            beta = 1-((Div_Bruta_por_PL-0.5)/(1-0.5))
        return beta
    
    ## Multiplicador para evitar investir em empresas com r_squared muito alto
    ## Se for maior que 0.5, o beta é 1
    ## Se for menor que 0.1, o beta é 0
    ## Se for entre 0.1 e 0.5, o beta é 1-((razao-0.1)/(0.5-0.1))
    def penalizacao_r_squared(r_squared):
        if r_squared>=0.5:
            beta = 1
        elif r_squared<=0.1:
            beta = 0
        else:

            beta = (r_squared-0.1)/(0.5-0.1)
        return beta
    

    ## Função beta    
    # Razao entre a expectativa da selic e o valor previsto pelo arima
    razao = prev_n_steps_meses/df_Expectativa_Selic_mensal.loc[df_Expectativa_Selic_mensal.index.max(),"Valor"]

    ## Mensurando a dívida
    Div_Bruta_por_PL = df_multiplos.loc[data_simulacao,"DIV_Bruta_PL"]

    # Retorno esperado anual
    Retorno_anual_esperado = Retorno_anual_esperado*\
        penalizacao_hike(razao)*penalizacao_divida(Div_Bruta_por_PL)*penalizacao_r_squared(r_squared)
    
    if Retorno_anual_esperado<0:
        Retorno_anual_esperado = 0
    
    Retorno_anual_esperado_percentual = round((Retorno_anual_esperado)*100,2)



    ## Coletando os dados para a lista de retorno

    ## Lista com os returns

    lista_return = [Setor , Retorno_anual_esperado, ROE, Div_Bruta_por_PL,
                    DY_medio, CAGR_Medio, Expansao_multiplo, Multiplo_atual, Multiplo_medio, Desvio_multiplo]



    return lista_return, lista_figuras


## Estratégia Baseada em Earnins Yield

def ret_PE(df_acao, df_multiplos, df_CAGR, df_Expectativa_Selic_mensal, prev_n_steps_meses):
    """
    Calcula a rentabilidade esperada de ativos com base no Earning Yields (Lucro/Preço), considerando penalizações para baixo ROE, crescimento e pagamento de dividendos.

    Parâmetros:
    df_acao (pd.DataFrame): DataFrame contendo dados fundamentalistas da ação normalizados pelo número de ações ex-tesouraria.
    df_multiplos (pd.DataFrame): DataFrame contendo os múltiplos da empresa (dados diários).
    df_CAGR (pd.DataFrame): DataFrame contendo a taxa de crescimento de dados fundamentalistas.
    df_Expectativa_Selic_mensal (pd.DataFrame): DataFrame contendo as expectativas mensais da taxa SELIC.
    prev_n_steps_meses (float): Valor previsto para a próxima expectativa da taxa SELIC.

    Retorna:
    lista_return (list): Lista de informações, incluindo setor, retorno anual esperado, ROE, dívida líquida/EBITDA, DY médio, CAGR médio, expansão do múltiplo, múltiplo atual, múltiplo médio e desvio do múltiplo.
    lista_figuras (list): Lista de figuras geradas durante a análise.

    A função calcula o retorno esperado dos ativos com base no Earning Yields (Lucro/Preço), considerando penalizações para baixo ROE, crescimento e pagamento de dividendos.

    Exemplo de uso:
    dados_return, figuras = ret_PE(df_acao, df_multiplos, df_CAGR, df_Expectativa_Selic_mensal, prev_n_steps_meses)
    """
    ## Coletando dados iniciais
    [data_simulacao, Setor, ROE, DY_medio, ticker, CAGR_Medio, 
     Multiplo_atual, Multiplo_medio, Desvio_multiplo, Expansao_Multiplo] = \
        Dados_iniciais(df_acao, df_multiplos, df_CAGR, "LL", "PE")
    lista_figuras = []

    # Retorno esperado anual
    if Multiplo_atual>0.1:
        Retorno_anual_esperado = 1/Multiplo_atual
    else:
        Retorno_anual_esperado = 0



    ## Agora é necessário realizar algumas penalizações, são elas:
    ## ROE -> Retorno sobre o patrimônio líquido
    ## Crescimento de lucro -> CAGR
    ## Dividend Yield -> DY



    ## Multiplicador para penalizar o ROE, se for menor que 0.05, o beta é 0
    ## Se for maior que 0.15, o beta é 1
    ## Se for entre 0.05 e 0.15, o beta é ((razao-0.05)/(0.15-0.05))
    def penalizacao_ROE(razao):
        if razao<=0.05:
            beta = 0
        elif razao>=0.15:
            beta = 1
        else:
            beta = ((razao-0.05)/(0.15-0.05))
        return beta
 
    ## Fim da função penalizacao_ROE

    ## A Penalização do CAGR Lucro será feita contabilizando o DY
    ## Multiplicador para penalizar o Crescimento,
    ## Se o (CAGR_Medio+1)*(DY_medio+1) for menor que 1, o beta é 0
    ## Se o (CAGR_Medio+1)*(DY_medio+1) for maior que 1.15, o beta é 1

    def penalizacao_Crescimento(CAGR_Medio,DY_medio):
        if (CAGR_Medio+1)*(DY_medio+1)<=1:
            beta = 0
        elif (CAGR_Medio+1)*(DY_medio+1)>=1.15:
            beta = 1
        else:
            beta = ((CAGR_Medio+1)*(DY_medio+1)-1)/(1.15-1)
        return beta
    
    ## Fim da função penalizacao_Crescimento
   
    ## Calculando as penalizações
    beta_ROE = penalizacao_ROE(ROE)
    beta_Crescimento = penalizacao_Crescimento(CAGR_Medio,DY_medio)

    ## Retorno esperado anual
    Retorno_anual_esperado = Retorno_anual_esperado*beta_ROE*beta_Crescimento

    ## Dívida Líquida / EBITDA
    Div_Liq_EBITDA = df_multiplos.loc[data_simulacao,"DIV_liq_EBITDA"]

    ## Lista com os returns
    lista_return = [Setor , Retorno_anual_esperado, ROE, Div_Liq_EBITDA,
                    DY_medio, CAGR_Medio, Expansao_Multiplo, Multiplo_atual, Multiplo_medio, Desvio_multiplo]
    

    return lista_return, lista_figuras


## Estratégia por EBITDA
def ret_EBITDA(df_acao, df_multiplos, df_CAGR, limite_ROE_inf, limite_ROE_sup, limite_divida):
    """
    Calcula a rentabilidade esperada de ativos baseada no crescimento do EBITDA, DY e expansão do múltiplo, com penalizações para baixo ROE e alta dívida líquida/EBITDA.

    Parâmetros:
    df_acao (pd.DataFrame): DataFrame contendo dados fundamentalistas da ação normalizados pelo número de ações ex-tesouraria.
    df_multiplos (pd.DataFrame): DataFrame contendo os múltiplos da empresa (dados diários).
    df_CAGR (pd.DataFrame): DataFrame contendo a taxa de crescimento de dados fundamentalistas.
    limite_ROE_inf (float): Limite inferior de ROE para penalização.
    limite_ROE_sup (float): Limite superior de ROE onde não há mais penalização.
    limite_divida (float): Valor a partir do qual começa a penalização de dívida líquida/EBITDA.

    Retorna:
    lista_return (list): Lista de informações, incluindo setor, retorno anual esperado, ROE, dívida líquida/EBITDA, DY médio, CAGR médio, expansão do múltiplo, múltiplo atual, múltiplo médio e desvio do múltiplo.
    lista_figuras (list): Lista de figuras geradas durante a análise.

    A função calcula o retorno esperado dos ativos com base no crescimento do EBITDA, DY e expansão do múltiplo, com penalizações para baixo ROE e alta dívida líquida/EBITDA.

    Exemplo de uso:
    dados_return, figuras = ret_EBITDA(df_acao, df_multiplos, df_CAGR, limite_ROE_inf, limite_ROE_sup, limite_divida)
    """
    ## Coletando dados iniciais
    [data_simulacao, Setor, ROE, DY_medio, ticker, CAGR_Medio, Multiplo_atual,
      Multiplo_medio, Desvio_multiplo, Expansao_Multiplo] = \
        Dados_iniciais(df_acao, df_multiplos, df_CAGR, "EBITDA", "EV_EBITDA")
    
    
    ## Lista de figuras
    lista_figuras = []

    ## Impressão dos dados
    

    # Retorno esperado anual
    ## Será composto por 2 partes:
    ## 1 - Regressão do EV/EBITDA a média histórica em x anos
    ## 2 - Crescimento do EBITDA somado ao DY médio

    ## Retorno esperado anual
    Retorno_anual_esperado = Expansao_Multiplo*(1+CAGR_Medio)*(1+DY_medio)-1

    ## Agora é necessário realizar algumas penalizações, são elas:
    ## ROE -> Retorno sobre o patrimônio líquido
    ## Dívida Líquida/EBITDA -> Dívida Líquida/EBITDA

    ## Pegar os dados de Dívida Líquida/EBITDA
    Divida_Liquida_EBITDA = df_multiplos.loc[data_simulacao,"DIV_liq_EBITDA"]

    ## Se o ROE for menor que limite_ROE_inf, o beta é 0
    ## Se o ROE for maior que limite_ROE_sup, o beta é 1

    ## Se o ROE for entre limite_ROE_inf e limite_ROE_sup, 
    # o beta é ((razao-limite_ROE_inf)/(limite_ROE_sup-limite_ROE_inf))
    def penalizacao_ROE(razao):
        if razao<=limite_ROE_inf:
            beta = 0
        elif razao>=limite_ROE_sup:
            beta = 1
        else:
            beta = ((razao-limite_ROE_inf)/(limite_ROE_sup-limite_ROE_inf))
        return beta
    
    ## Se a Dívida Líquida/EBITDA for menor que limite_divida, o beta é 1
    ## Se a Dívida Líquida/EBITDA for maior que 2*limite_divida, o beta é 0
    ## Se a Dívida Líquida/EBITDA for entre limite_divida e 2*limite_divida, o beta é 1-((razao-limite_divida)/(limite_divida))
    def penalizacao_Divida_Liquida_EBITDA(razao):
        if razao<=limite_divida:
            beta = 1
        elif razao>=2*limite_divida:
            beta = 0
        else:
            beta = 1-((razao-limite_divida)/(limite_divida))
        return beta
    
    ## Calculando as penalizações
    beta_ROE = penalizacao_ROE(ROE)
    beta_Divida_Liquida_EBITDA = penalizacao_Divida_Liquida_EBITDA(Divida_Liquida_EBITDA)

    ## Retorno esperado anual
    Retorno_anual_esperado = Retorno_anual_esperado*beta_ROE*beta_Divida_Liquida_EBITDA

    if Retorno_anual_esperado<0:
        Retorno_anual_esperado = 0

    ## Lista com os returns
    lista_return = [Setor, Retorno_anual_esperado, ROE, Divida_Liquida_EBITDA,
                    DY_medio, CAGR_Medio, Expansao_Multiplo, Multiplo_atual, Multiplo_medio, Desvio_multiplo]



    return lista_return, lista_figuras 

## main retorno

def main_ret(dict_df_acoes, df_Selic, df_Expectativa_Selic_mensal):
    """
    Calcula a rentabilidade esperada de ativos com diferentes estratégias dependendo do setor.

    Parâmetros:
    dict_df_acoes (dict): Dicionário com chaves sendo os ativos e valores contendo (df_acao, df_multiplos, df_CAGR).
    df_Selic (pd.DataFrame): DataFrame com as taxas SELIC.
    df_Expectativa_Selic_mensal (pd.DataFrame): DataFrame com a expectativa da SELIC mensal.

    Retorna:
    info_main (list): Lista contendo informações para o processo principal, incluindo o DataFrame de retorno esperado, cotações ajustadas e setores.
    info_verificao (list): Lista contendo informações para verificação, incluindo o DataFrame de retorno, cotações ajustadas, figuras e lista de excluídos.

    A função calcula o retorno esperado dos ativos com diferentes estratégias dependendo do setor em que estão inseridos. Também fornece informações para verificação e informações essenciais para o processo principal.

    Exemplo de uso:
    info_main, info_verificao = main_ret(dict_df_acoes, df_Selic, df_Expectativa_Selic_mensal)
    """
    ## Lista de ativos elegíveis
    lista_ativos_elegiveis = list(dict_df_acoes.keys())
    lista_ativos_elegiveis.sort()

    ## Obter a previsão da expectativa da SELIC
    prev_n_steps_meses = df_Expectativa_Selic_mensal.loc[df_Expectativa_Selic_mensal.index.max(),"1_mes"]

    ## Criando o dataframe de retorno
    df_retorno = pd.DataFrame(columns=[
        "Setor","Retorno_anual_esperado", "ROE", "Multiplo_Divida",
        "DY_medio", "CAGR_Medio", "Expansao_Multiplo", "Multiplo_atual","Multiplo_medio","Desvio_multiplo",])
    df_cotacao_ajustado = pd.DataFrame()
    dic_figuras = {}
    lista_excluidos = []
    for ativo in lista_ativos_elegiveis:
        ## Pegando os dados dos ativos
        [df_acao, df_multiplos, df_CAGR] = dict_df_acoes[ativo]
        Setor = df_acao.loc[:,"Setor_Comdinheiro"][0]
        try:
            if Setor in ["Construção e Imóveis"]:
                ## Retorno esperado anual
                lista_return, lista_figuras = \
                ret_VPA(df_acao, df_multiplos, df_CAGR, df_Expectativa_Selic_mensal, prev_n_steps_meses)

                ## Adicionando o retorno esperado
                df_retorno.loc[ativo,:] = lista_return
                dic_figuras[ativo] = lista_figuras

                ## Adicionando a cotação ajustada
                df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
                df_ativo.columns = [ativo]
                df_cotacao_ajustado = pd.concat([df_cotacao_ajustado, df_ativo], axis=1, join="outer")


            elif Setor in ["Bancos e Serviços Financeiros"]:
                ## Retorno esperado anual
                lista_return, lista_figuras = \
                ret_PE(df_acao, df_multiplos, df_CAGR, df_Expectativa_Selic_mensal, prev_n_steps_meses)
                ## Adicionando o retorno esperado
                df_retorno.loc[ativo,:] = lista_return
                dic_figuras[ativo] = lista_figuras

                ## Adicionando a cotação ajustada
                df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
                df_ativo.columns = [ativo]
                df_cotacao_ajustado = pd.concat([df_cotacao_ajustado, df_ativo], axis=1, join="outer")


            elif Setor in ["Energia e Serviços Básicos"]:
                ## Retorno esperado anual
                lista_return, lista_figuras = \
                ret_EBITDA(df_acao, df_multiplos, df_CAGR, 0.02 , 0.1, 3)

                ## Adicionando o retorno esperado
                df_retorno.loc[ativo,:] = lista_return
                dic_figuras[ativo] = lista_figuras

                ## Adicionando a cotação ajustada
                df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
                df_ativo.columns = [ativo]
                df_cotacao_ajustado = pd.concat([df_cotacao_ajustado, df_ativo], axis=1, join="outer")

            elif Setor in ["Biocombustíveis, Gás e Petróleo", "Mineração"]:
                ## Retorno esperado anual
                lista_return, lista_figuras = \
                ret_EBITDA(df_acao, df_multiplos, df_CAGR, 0.15 , 0.3, 1.5)

                ## Adicionando o retorno esperado
                df_retorno.loc[ativo,:] = lista_return
                dic_figuras[ativo] = lista_figuras

                ## Adicionando a cotação ajustada
                df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
                df_ativo.columns = [ativo]
                df_cotacao_ajustado = pd.concat([df_cotacao_ajustado, df_ativo], axis=1, join="outer")

            elif Setor in ["Serviços", "Celulose, Papel e Madeira", "Indústria"]:
                ## Retorno esperado anual
                lista_return, lista_figuras = \
                ret_EBITDA(df_acao, df_multiplos, df_CAGR, 0.01 , 0.05, 2.5)

                ## Adicionando o retorno esperado
                df_retorno.loc[ativo,:] = lista_return
                dic_figuras[ativo] = lista_figuras

                ## Adicionando a cotação ajustada
                df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
                df_ativo.columns = [ativo]
                df_cotacao_ajustado = pd.concat([df_cotacao_ajustado, df_ativo], axis=1, join="outer")
    
        except:
            print(f"Ticker: {ativo}, Setor: {Setor} foi excluído.")
            raise
           
    
    ## Informações para verificação
    df_cotacao_ajustado.dropna(how="all",inplace=True)
    info_verificao = [df_retorno.copy(), df_cotacao_ajustado.copy(), dic_figuras, lista_excluidos]

    ## Informações para o main
    # Retirar ativos iguais e com rentabilidade esperada menor que 5%

    df_retorno_new = df_retorno.sort_values(by='Retorno_anual_esperado', ascending=False).copy()
    # Extraia os primeiros quatro caracteres do índice para criar uma nova coluna 'Ticker'
    df_retorno_new['Ativo'] = df_retorno_new.index.str[:4]

    # Defina o índice como 'Ativo'
    df_retorno_new.reset_index(inplace=True)
    df_retorno_new.set_index('Ativo', inplace=True)

    # Use o método groupby para agrupar por ativo e manter a primeira ocorrência de cada grupo (com o maior retorno)
    df_retorno_new = df_retorno_new.groupby(df_retorno_new.index).first()
    df_retorno_new.set_index('index', inplace=True)
    
    
    # Retirar ativos com rentabilidade esperada menor que 5%
    df_retorno_new = df_retorno_new[df_retorno_new["Retorno_anual_esperado"]>=0.05]
    df_retorno_new.sort_index(inplace=True)

    # Redefinir o df_cotacao_ajustado
    ativos = df_retorno_new.index
    df_cot_new = df_cotacao_ajustado.loc[:,ativos].copy()
    ## Filtrar os últimos 4 anos
    df_cot_new = df_cot_new.iloc[-4*252:-5,:] # últimos 4 anos

    ## Definir as informações que serão utilizadas para a montagem do portfólio
    info_main = [df_retorno_new["Retorno_anual_esperado"], df_cot_new, df_retorno_new["Setor"]]
    

    return info_main, info_verificao





