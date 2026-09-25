# -*- coding: utf-8 -*-
"""
Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs
==================================================================

Entrega 4 do desafio de NLP Aplicado.

COMO RODAR
----------
    pip install -r requirements.txt
    streamlit run app_ouvidoria.py

O app abre no navegador em http://localhost:8501

O QUE ELE FAZ
-------------
A Ouvidoria recebe cerca de 4.000 manifestações por mês em texto livre. O
sistema atual usa busca por palavra-chave, que não encontra duas reclamações
sobre o mesmo problema quando o cidadão usa palavras diferentes. Este
protótipo troca a busca textual por **similaridade semântica**: cada
manifestação vira um vetor numérico que representa o seu significado, e a
comparação passa a ser entre significados, não entre palavras.

ABAS
----
    🔍 Busca Semântica  — descrição livre -> manifestações mais parecidas
    📋 Base Completa    — tabela + matriz de similaridade
    🌐 Espaço Vetorial  — projeção 2D (PCA / t-SNE) por categoria
    🧩 Chunking         — divisão de textos longos em pedaços

DECISÕES DE IMPLEMENTAÇÃO
-------------------------
* `st.cache_resource` guarda o MODELO (objeto pesado, não serializável).
  `st.cache_data` guarda os EMBEDDINGS (dados, serializáveis). Essa é a
  divisão que a documentação do Streamlit recomenda e que o enunciado pede.
* Os vetores são normalizados para tamanho 1, o que faz o produto escalar
  ser igual ao cosseno. A matriz 40x40 sai de uma multiplicação só.
* Três modelos ficam disponíveis na barra lateral para comparação.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ===========================================================================
# CONFIGURAÇÃO GERAL
# ===========================================================================

st.set_page_config(
    page_title="Ouvidoria Inteligente",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAIZ = Path(__file__).resolve().parent

# --- Faixas de score definidas no enunciado --------------------------------
# 🟢 acima de 0.7 | 🟡 acima de 0.5 | 🔴 as demais
LIMITE_VERDE = 0.70
LIMITE_AMARELO = 0.50

CORES = {
    "verde": "#16a34a",
    "amarelo": "#d97706",
    "vermelho": "#dc2626",
}

# --- Modelos disponíveis ---------------------------------------------------
# O enunciado pede, nas dicas, comparar pelo menos dois modelos. A barra
# lateral permite trocar e ver o efeito em todas as abas.
MODELOS = {
    "MiniLM (rápido, padrão do time)": {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "tamanho": "~470 MB",
        "nota": "Usado nas Entregas 1, 2 e 3. Bom equilíbrio entre "
                "velocidade e qualidade.",
    },
    "MPNet (maior, mais preciso)": {
        "id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "tamanho": "~1,1 GB",
        "nota": "Irmão maior do MiniLM, treinado para detectar paráfrase. "
                "Separa melhor os pares, mas ocupa o dobro de memória.",
    },
    "E5-small (otimizado para busca)": {
        "id": "intfloat/multilingual-e5-small",
        "dim": 384,
        "tamanho": "~470 MB",
        "nota": "Treinado especificamente para recuperação. Exige prefixos "
                "'query:' e 'passage:' — o app aplica automaticamente.",
    },
}

# --- Corpora disponíveis ---------------------------------------------------
# O repositório do time tem dois arquivos. Deixar a escolha visível evita
# que alguém compare números calculados sobre bases diferentes sem perceber.
CORPORA = {
    "manifestacoes_exemplo.json": "Base usada nas Entregas 1 e 2.",
    "manifestacoes.json": "Base usada na Entrega 3. Atenção: cinco registros "
                          "(M003, M008, M017, M022, M031) têm menos de 50 "
                          "caracteres, abaixo do mínimo que o enunciado define.",
}

# --- Estilos ---------------------------------------------------------------
st.markdown(
    """
    <style>
      /* Cartão de resultado da busca */
      .cartao {
          border: 1px solid #e2e8f0;
          border-left: 5px solid var(--cor);
          border-radius: 8px;
          padding: 14px 18px;
          margin-bottom: 12px;
          background: #ffffff;
      }
      .cartao-cabecalho {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
      }
      .cartao-id {
          font-weight: 700;
          font-size: 0.95rem;
          color: #0f172a;
      }
      .cartao-score {
          font-weight: 700;
          font-size: 1.05rem;
          color: var(--cor);
      }
      .cartao-meta {
          font-size: 0.78rem;
          color: #64748b;
          margin-bottom: 8px;
      }
      .cartao-texto {
          font-size: 0.92rem;
          line-height: 1.5;
          color: #1e293b;
      }
      .barra-fundo {
          background: #f1f5f9;
          border-radius: 4px;
          height: 7px;
          margin-top: 10px;
          overflow: hidden;
      }
      .barra-preenchida {
          background: var(--cor);
          height: 100%;
          border-radius: 4px;
      }
      /* Deixa o texto das abas um pouco maior */
      button[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# CARREGAMENTO (com cache)
# ===========================================================================

@st.cache_data(show_spinner=False)
def carregar_corpus(nome_arquivo: str) -> pd.DataFrame:
    """Lê o JSON de manifestações.

    Usa `st.cache_data` porque o retorno é um DataFrame, ou seja, DADOS.
    O Streamlit serializa e reaproveita entre execuções.
    """
    caminho = RAIZ / nome_arquivo
    if not caminho.exists():
        return pd.DataFrame()
    registros = json.loads(caminho.read_text(encoding="utf-8"))
    df = pd.DataFrame(registros)
    df["n_chars"] = df["texto"].str.len()
    return df


@st.cache_resource(show_spinner=False)
def carregar_modelo(id_modelo: str) -> SentenceTransformer:
    """Carrega o modelo de embedding.

    Usa `st.cache_resource` porque um SentenceTransformer é um OBJETO pesado
    e não serializável. O cache mantém uma única instância viva e
    compartilhada, em vez de recarregar a cada interação do usuário.
    """
    return SentenceTransformer(id_modelo)


def _prefixo_e5(id_modelo: str, textos: list[str], papel: str) -> list[str]:
    """Aplica os prefixos que a família E5 exige.

    O E5 foi treinado com marcadores explícitos: o texto do acervo leva
    'passage: ' e a consulta do usuário leva 'query: '. Sem isso o modelo
    roda e não reclama, mas entrega qualidade abaixo da anunciada — é um
    erro silencioso.
    """
    if "e5" not in id_modelo.lower():
        return textos
    return [f"{papel}: {t}" for t in textos]


@st.cache_data(show_spinner=False)
def gerar_embeddings(id_modelo: str, textos: tuple[str, ...],
                     papel: str = "passage") -> np.ndarray:
    """Converte textos em vetores normalizados.

    `st.cache_data` guarda o resultado (um array numpy = dados). A chave do
    cache inclui o id do modelo e os textos, então trocar de modelo ou de
    corpus recalcula corretamente.

    Os textos chegam como tupla porque o cache exige argumentos "hasháveis"
    — uma lista não serve.
    """
    modelo = carregar_modelo(id_modelo)
    entrada = _prefixo_e5(id_modelo, list(textos), papel)
    return modelo.encode(entrada, normalize_embeddings=True,
                         show_progress_bar=False)


# ===========================================================================
# FUNÇÕES DE APOIO
# ===========================================================================

def faixa_de(score: float) -> tuple[str, str, str]:
    """Classifica um score nas três faixas do enunciado.

    Retorna (nome, emoji, cor em hexadecimal).
    """
    if score > LIMITE_VERDE:
        return "verde", "🟢", CORES["verde"]
    if score > LIMITE_AMARELO:
        return "amarelo", "🟡", CORES["amarelo"]
    return "vermelho", "🔴", CORES["vermelho"]


def buscar(consulta: str, df: pd.DataFrame, emb_corpus: np.ndarray,
           id_modelo: str, k: int) -> pd.DataFrame:
    """Busca semântica: devolve as k manifestações mais parecidas.

    O texto do usuário vira um vetor no MESMO espaço das manifestações, e a
    comparação é o cosseno entre eles. Como todos os vetores estão
    normalizados, basta um produto escalar.
    """
    vetor = gerar_embeddings(id_modelo, (consulta,), papel="query")[0]
    scores = emb_corpus @ vetor              # similaridade com cada manifestação
    ordem = np.argsort(-scores)[:k]          # as k maiores, em ordem

    resultado = df.iloc[ordem].copy()
    resultado["score"] = scores[ordem]
    return resultado.reset_index(drop=True)


def matriz_similaridade(emb: np.ndarray) -> np.ndarray:
    """Matriz N x N com a similaridade de cada par."""
    return emb @ emb.T


def cartao_resultado(linha: pd.Series, posicao: int) -> str:
    """Monta o HTML de um cartão de resultado da busca."""
    _, emoji, cor = faixa_de(linha["score"])
    largura = max(0.0, min(1.0, float(linha["score"]))) * 100
    texto = (linha["texto"].replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;"))
    return f"""
    <div class="cartao" style="--cor:{cor}">
      <div class="cartao-cabecalho">
        <span class="cartao-id">{emoji} &nbsp;{posicao}º &nbsp;·&nbsp; {linha['id']}</span>
        <span class="cartao-score">{linha['score']:.4f}</span>
      </div>
      <div class="cartao-meta">{linha['categoria_oficial']} &nbsp;·&nbsp;
        {linha['data']} &nbsp;·&nbsp; {linha['n_chars']} caracteres</div>
      <div class="cartao-texto">{texto}</div>
      <div class="barra-fundo"><div class="barra-preenchida"
           style="width:{largura:.1f}%"></div></div>
    </div>
    """


# ===========================================================================
# BARRA LATERAL
# ===========================================================================

with st.sidebar:
    st.markdown("## 🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs")
    st.divider()

    st.markdown("### Base de dados")
    nome_corpus = st.selectbox(
        "Arquivo de manifestações",
        options=list(CORPORA.keys()),
        index=0,
        help="O repositório tem dois arquivos. Escolher aqui evita comparar "
             "números calculados sobre bases diferentes.",
    )
    st.caption(CORPORA[nome_corpus])

    df = carregar_corpus(nome_corpus)
    if df.empty:
        st.error(f"Arquivo `{nome_corpus}` não encontrado na pasta do app.")
        st.stop()

    st.divider()
    st.markdown("### Modelo de embedding")
    nome_modelo = st.selectbox(
        "Modelo",
        options=list(MODELOS.keys()),
        index=0,
        help="Modelos multilíngues com suporte a português. Trocar aqui "
             "recalcula todas as abas.",
    )
    info_modelo = MODELOS[nome_modelo]
    st.caption(info_modelo["nota"])

    st.divider()
    st.markdown("### Busca")
    top_k = st.slider(
        "Quantos resultados retornar (top-k)",
        min_value=1, max_value=15, value=5,
        help="O enunciado pede 5 por padrão.",
    )

    st.divider()
    with st.expander("ℹ️ Faixas de score"):
        st.markdown(
            f"""
            - 🟢 **acima de {LIMITE_VERDE}** — muito provavelmente o mesmo assunto
            - 🟡 **acima de {LIMITE_AMARELO}** — relacionado, vale conferir
            - 🔴 **abaixo disso** — provavelmente sem relação

            Estas faixas são as definidas no enunciado. Elas coincidem com a
            triagem adotada na Entrega 2: agrupar no verde, revisar no
            amarelo, separar no vermelho.
            """
        )

# --- Carrega modelo e embeddings uma única vez -----------------------------
id_modelo = info_modelo["id"]
TEXTOS = tuple(df["texto"].tolist())

with st.spinner(f"Carregando **{nome_modelo}** e calculando embeddings… "
                f"(na primeira vez baixa {info_modelo['tamanho']})"):
    modelo = carregar_modelo(id_modelo)
    EMB = gerar_embeddings(id_modelo, TEXTOS, papel="passage")

IDS = df["id"].tolist()
LIMITE_TOKENS = modelo.max_seq_length

with st.sidebar:
    st.divider()
    st.markdown("### Modelo carregado")
    c1, c2 = st.columns(2)
    c1.metric("Dimensões", EMB.shape[1])
    c2.metric("Limite", f"{LIMITE_TOKENS} tok")

    # Quantos textos o modelo está cortando?
    n_tok = [len(modelo.tokenizer.encode(t, add_special_tokens=True))
             for t in TEXTOS]
    truncados = sum(1 for n in n_tok if n > LIMITE_TOKENS)
    if truncados:
        st.warning(
            f"**{truncados} de {len(df)} manifestações são truncadas.** "
            f"O modelo lê no máximo {LIMITE_TOKENS} tokens e descarta o resto "
            f"sem avisar. Ver a aba 🧩 Chunking.",
            icon="⚠️",
        )

# ===========================================================================
# CABEÇALHO
# ===========================================================================

st.title("🏛️ Ouvidoria Inteligente")
st.markdown(
    "Busca **por significado**, não por palavra-chave. Duas pessoas que "
    "relatam o mesmo buraco com palavras diferentes passam a cair no mesmo "
    "lugar."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Manifestações", len(df))
col2.metric("Categorias", df["categoria_oficial"].nunique())
col3.metric("Pares possíveis", f"{len(df) * (len(df) - 1) // 2}")
col4.metric("Texto médio", f"{df.n_chars.mean():.0f} car.")

st.divider()

aba_busca, aba_base, aba_espaco, aba_chunk = st.tabs([
    "🔍 Busca Semântica",
    "📋 Base Completa",
    "🌐 Espaço Vetorial",
    "🧩 Chunking",
])

# ===========================================================================
# ABA 1 — BUSCA SEMÂNTICA
# ===========================================================================

with aba_busca:
    st.subheader("Buscar manifestações parecidas")
    st.markdown(
        "Descreva um problema com as suas próprias palavras. O sistema "
        f"devolve as **{top_k} manifestações mais próximas em significado** — "
        "mesmo que não compartilhem nenhuma palavra com o que você escreveu."
    )

    # Exemplos prontos: ajudam quem abre o app pela primeira vez e não sabe o
    # que digitar. Foram escolhidos por MEDIÇÃO, não por intuição — cada um
    # traz o primeiro resultado na faixa verde no corpus de exemplo, e
    # nenhum repete o vocabulário da manifestação que encontra. É isso que
    # demonstra a busca semântica: "crateras" acha "esburacado", "creche
    # quente" acha "ar-condicionado quebrado".
    EXEMPLOS = {
        "🕳️ Via destruída": "a rua está cheia de crateras e já estraguei o pneu do carro",
        "💊 Falta remédio": "faltou remédio de pressão na farmácia do posto",
        "💡 Rua escura": "a iluminação da rua está apagada e fica tudo escuro à noite",
        "🏫 Creche quente": "a creche está com o ar-condicionado quebrado e as crianças passam calor",
    }

    st.caption("Ou experimente um exemplo:")
    colunas_ex = st.columns(len(EXEMPLOS))
    for coluna, (rotulo, texto_exemplo) in zip(colunas_ex, EXEMPLOS.items()):
        if coluna.button(rotulo, use_container_width=True):
            st.session_state["consulta"] = texto_exemplo

    consulta = st.text_area(
        "Descrição do problema",
        key="consulta",
        height=110,
        placeholder="Ex.: o asfalto da avenida está destruído e cheio de buracos…",
        label_visibility="collapsed",
    )

    if consulta and consulta.strip():
        resultados = buscar(consulta, df, EMB, id_modelo, top_k)

        # Resumo por faixa, para o usuário entender a qualidade do resultado
        faixas = [faixa_de(s)[0] for s in resultados["score"]]
        n_verde = faixas.count("verde")
        n_amarelo = faixas.count("amarelo")
        n_vermelho = faixas.count("vermelho")

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Forte", n_verde, help=f"score acima de {LIMITE_VERDE}")
        c2.metric("🟡 Possível", n_amarelo, help=f"entre {LIMITE_AMARELO} e {LIMITE_VERDE}")
        c3.metric("🔴 Fraco", n_vermelho, help=f"abaixo de {LIMITE_AMARELO}")

        if n_verde == 0 and n_amarelo == 0:
            st.info(
                "Nenhum resultado passou de 0.5. Provavelmente não existe "
                "manifestação parecida na base — o que, numa Ouvidoria, "
                "significa que esta seria uma demanda nova.",
                icon="💡",
            )

        st.markdown("")
        for i, linha in resultados.iterrows():
            st.markdown(cartao_resultado(linha, i + 1), unsafe_allow_html=True)

        with st.expander("📊 Ver os scores em gráfico"):
            fig = go.Figure(go.Bar(
                x=resultados["score"],
                y=[f"{r.id}" for r in resultados.itertuples()],
                orientation="h",
                marker_color=[faixa_de(s)[2] for s in resultados["score"]],
                text=[f"{s:.4f}" for s in resultados["score"]],
                textposition="auto",
                hovertext=[t[:120] + "…" for t in resultados["texto"]],
            ))
            fig.add_vline(x=LIMITE_VERDE, line_dash="dash",
                          line_color=CORES["verde"], annotation_text="0.70")
            fig.add_vline(x=LIMITE_AMARELO, line_dash="dash",
                          line_color=CORES["amarelo"], annotation_text="0.50")
            fig.update_layout(
                height=60 + 38 * len(resultados),
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="similaridade de cosseno",
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Digite uma descrição acima ou clique em um dos exemplos.",
                icon="👆")

# ===========================================================================
# ABA 2 — BASE COMPLETA
# ===========================================================================

with aba_base:
    st.subheader("Todas as manifestações")

    c1, c2 = st.columns([2, 3])
    categorias = c1.multiselect(
        "Filtrar por categoria",
        options=sorted(df["categoria_oficial"].unique()),
        default=[],
        placeholder="todas as categorias",
    )
    termo = c2.text_input(
        "Filtrar por palavra no texto",
        placeholder="ex.: buraco, posto, escola…",
    )

    visao = df.copy()
    if categorias:
        visao = visao[visao["categoria_oficial"].isin(categorias)]
    if termo.strip():
        visao = visao[visao["texto"].str.contains(termo.strip(), case=False,
                                                  na=False)]

    st.caption(f"Mostrando **{len(visao)}** de {len(df)} manifestações")
    st.dataframe(
        visao[["id", "data", "categoria_oficial", "n_chars", "texto"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "data": st.column_config.TextColumn("Data", width="small"),
            "categoria_oficial": st.column_config.TextColumn("Categoria",
                                                             width="medium"),
            "n_chars": st.column_config.NumberColumn("Caracteres",
                                                     width="small"),
            "texto": st.column_config.TextColumn("Texto", width="large"),
        },
        height=420,
    )

    st.divider()
    st.markdown("### Matriz de similaridade")
    st.markdown(
        "Cada célula é a similaridade entre duas manifestações. A diagonal "
        "foi removida — todo texto é idêntico a si mesmo, e deixá-la colorida "
        "distorce a escala e esconde o que interessa."
    )

    if st.button("🔢 Gerar matriz de similaridade", type="primary"):
        S = matriz_similaridade(EMB)
        S_plot = S.copy()
        np.fill_diagonal(S_plot, np.nan)      # esconde a diagonal

        fig = px.imshow(
            S_plot,
            x=IDS, y=IDS,
            color_continuous_scale="RdYlGn_r",
            aspect="equal",
            labels=dict(color="similaridade"),
        )
        fig.update_layout(
            height=760,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(tickfont=dict(size=8), side="bottom"),
            yaxis=dict(tickfont=dict(size=8)),
        )
        fig.update_traces(
            hovertemplate="%{y} × %{x}<br>similaridade: %{z:.4f}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Ranking dos pares mais parecidos: é o que a Ouvidoria olharia
        # primeiro para encontrar duplicatas.
        pares = []
        for a, b in itertools.combinations(range(len(IDS)), 2):
            pares.append({
                "Par": f"{IDS[a]} × {IDS[b]}",
                "Similaridade": round(float(S[a, b]), 4),
                "Faixa": faixa_de(S[a, b])[1],
                "Categorias": (f"{df.categoria_oficial.iloc[a]} / "
                               f"{df.categoria_oficial.iloc[b]}"),
            })
        pares = pd.DataFrame(pares).sort_values("Similaridade",
                                                ascending=False)

        st.markdown("#### Os 15 pares mais parecidos")
        st.caption(
            "Candidatos a duplicata. Na Entrega 2 este ranking foi a base do "
            "detector — os pares no topo são os que a Ouvidoria deveria "
            "conferir primeiro."
        )
        st.dataframe(pares.head(15), use_container_width=True,
                     hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Pares acima de 0.70", int((pares.Similaridade > LIMITE_VERDE).sum()))
        c2.metric("🟡 Entre 0.50 e 0.70",
                  int(((pares.Similaridade > LIMITE_AMARELO) &
                       (pares.Similaridade <= LIMITE_VERDE)).sum()))
        c3.metric("Maior similaridade", f"{pares.Similaridade.max():.4f}")

# ===========================================================================
# ABA 3 — ESPAÇO VETORIAL
# ===========================================================================

with aba_espaco:
    st.subheader("O espaço semântico em 2D")
    st.markdown(
        "Cada manifestação é um ponto. Os embeddings têm centenas de "
        "dimensões, então é preciso **projetá-los em 2D** para enxergar. "
        "Pontos próximos significam textos com sentido parecido."
    )

    c1, c2 = st.columns([1, 2])
    tecnica = c1.radio(
        "Técnica de projeção",
        options=["PCA", "t-SNE"],
        horizontal=True,
        help="PCA preserva a estrutura global e é determinístico. "
             "t-SNE preserva vizinhanças locais e separa grupos melhor, "
             "mas as distâncias entre grupos não têm significado.",
    )

    if tecnica == "t-SNE":
        perplexidade = c2.slider(
            "Perplexidade",
            min_value=2, max_value=max(3, len(df) // 2), value=min(12, len(df) // 3),
            help="Quantos vizinhos cada ponto considera. Valores baixos "
                 "destacam grupos pequenos; altos, a estrutura geral. "
                 "Precisa ser menor que o número de manifestações.",
        )

    @st.cache_data(show_spinner=False)
    def projetar(emb: np.ndarray, tecnica: str, perp: int = 12) -> np.ndarray:
        """Reduz os embeddings para 2 dimensões.

        Em cache porque o t-SNE é lento e seria refeito a cada clique.
        """
        if tecnica == "PCA":
            return PCA(n_components=2, random_state=42).fit_transform(emb)
        return TSNE(n_components=2, random_state=42, perplexity=perp,
                    init="pca", max_iter=1000).fit_transform(emb)

    with st.spinner(f"Projetando com {tecnica}…"):
        coords = projetar(EMB, tecnica,
                          perplexidade if tecnica == "t-SNE" else 12)

    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["resumo"] = plot_df["texto"].str.slice(0, 110) + "…"

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="categoria_oficial",
        text="id",
        hover_data={"x": False, "y": False, "id": True,
                    "categoria_oficial": True, "resumo": True},
        labels={"categoria_oficial": "Categoria oficial"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="top center",
                      textfont=dict(size=8, color="#475569"),
                      marker=dict(size=13, line=dict(width=1, color="white")))
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=f"{tecnica} — componente 1",
        yaxis_title=f"{tecnica} — componente 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # O enunciado pede explicitamente: "o aluno deve comentar se os clusters
    # semânticos coincidem com as categorias". A resposta abaixo é calculada,
    # não escrita à mão, e se atualiza quando o modelo ou o corpus muda.
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### Os clusters coincidem com as categorias oficiais?")

    # Medida 1: similaridade média dentro da mesma categoria x entre categorias
    S = matriz_similaridade(EMB)
    cats = df["categoria_oficial"].values
    dentro, entre = [], []
    for a, b in itertools.combinations(range(len(df)), 2):
        (dentro if cats[a] == cats[b] else entre).append(float(S[a, b]))
    media_dentro, media_entre = float(np.mean(dentro)), float(np.mean(entre))
    diferenca = media_dentro - media_entre

    # Medida 2: silhueta sobre os embeddings completos, usando a categoria
    # oficial como rótulo. Vai de -1 a 1: acima de 0 indica que os textos da
    # mesma categoria estão, em média, mais próximos entre si.
    try:
        silhueta = float(silhouette_score(EMB, cats, metric="cosine"))
    except Exception:
        silhueta = float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dentro da mesma categoria", f"{media_dentro:.4f}",
              help="similaridade média entre pares da mesma categoria")
    c2.metric("Entre categorias diferentes", f"{media_entre:.4f}",
              help="similaridade média entre pares de categorias distintas")
    c3.metric("Diferença", f"{diferenca:+.4f}",
              delta=f"{diferenca / media_entre * 100:+.0f}%")

    if diferenca > 0.05:
        st.success(
            f"**Sim, parcialmente.** Manifestações da mesma categoria são, em "
            f"média, **{diferenca:.4f} mais parecidas** entre si "
            f"({media_dentro:.4f}) do que com as de outras categorias "
            f"({media_entre:.4f}). O modelo captou a organização temática da "
            f"Ouvidoria **sem nunca ter visto os rótulos** — ele só leu os "
            f"textos.",
            icon="✅",
        )
    else:
        st.warning(
            f"**Pouca correspondência.** A diferença entre dentro e fora da "
            f"categoria é de apenas {diferenca:.4f}, o que indica que o "
            f"espaço vetorial não reflete bem as categorias oficiais neste "
            f"corpus.",
            icon="⚠️",
        )

    st.markdown(
        f"""
        **Mas a coincidência não é perfeita, e isso é esperado.** O
        coeficiente de silhueta é **{silhueta:.3f}** (a escala vai de −1 a 1;
        próximo de 0 significa fronteiras difusas).

        Dois motivos explicam a diferença:

        1. **As categorias são administrativas, o espaço é semântico.**
           "Falta de iluminação numa travessa escura" pode ser classificada
           como *infraestrutura* pela Ouvidoria e como *segurança* pelo
           cidadão que a escreveu. O modelo coloca esse texto entre os dois
           grupos, porque é onde ele de fato está.

        2. **Os agrupamentos que o modelo forma são mais finos que as
           categorias.** Dentro de *saúde* ele separa "falta de medicamento"
           de "demora no atendimento" — uma distinção que a categoria oficial
           não faz, mas que é útil para a triagem.

        **Conclusão prática:** o espaço vetorial não substitui a
        categorização oficial, mas resolve o problema (2) do enunciado —
        agrupar temas relacionados que hoje ficam dispersos.
        """
    )

    with st.expander("🔬 Detalhe por categoria"):
        linhas = []
        for cat in sorted(df["categoria_oficial"].unique()):
            idx = np.where(cats == cat)[0]
            if len(idx) < 2:
                continue
            internos = [float(S[a, b]) for a, b in itertools.combinations(idx, 2)]
            externos = [float(S[a, b]) for a in idx
                        for b in range(len(df)) if cats[b] != cat]
            linhas.append({
                "Categoria": cat,
                "Manifestações": len(idx),
                "Coesão interna": round(float(np.mean(internos)), 4),
                "Similaridade externa": round(float(np.mean(externos)), 4),
                "Separação": round(float(np.mean(internos) - np.mean(externos)), 4),
            })
        detalhe = pd.DataFrame(linhas).sort_values("Separação", ascending=False)
        st.dataframe(detalhe, use_container_width=True, hide_index=True)
        st.caption(
            "Separação alta = categoria bem definida no espaço vetorial. "
            "Separação baixa = os textos dessa categoria se misturam com os "
            "das outras."
        )

# ===========================================================================
# ABA 4 — CHUNKING
# ===========================================================================

with aba_chunk:
    st.subheader("Dividir manifestações longas em pedaços")
    st.markdown(
        f"""
        O modelo lê no máximo **{LIMITE_TOKENS} tokens** (cerca de 400 a 500
        caracteres em português) e **descarta o resto sem avisar**. Uma
        manifestação longa acaba representada só pelo seu começo.

        O *chunking* resolve isso dividindo o texto em pedaços que cabem no
        limite, cada um com o seu próprio vetor.
        """
    )

    longas = df.nlargest(5, "n_chars")
    escolha = st.selectbox(
        "Carregar uma manifestação longa do corpus",
        options=["(colar texto manualmente)"] +
                [f"{r.id} — {r.categoria_oficial} — {r.n_chars} caracteres"
                 for r in longas.itertuples()],
    )

    texto_inicial = ""
    if escolha != "(colar texto manualmente)":
        id_escolhido = escolha.split(" — ")[0]
        texto_inicial = df.loc[df["id"] == id_escolhido, "texto"].iloc[0]

    texto_longo = st.text_area(
        "Texto da manifestação",
        value=texto_inicial,
        height=200,
        placeholder="Cole aqui uma manifestação longa…",
    )

    st.markdown("#### Parâmetros da divisão")
    c1, c2, c3 = st.columns(3)
    estrategia = c1.selectbox(
        "Estratégia",
        options=["RecursiveCharacterTextSplitter", "CharacterTextSplitter"],
        help="A recursiva tenta cortar em parágrafo, depois frase, depois "
             "vírgula, depois espaço — preservando a estrutura. A simples "
             "corta sempre no mesmo separador.",
    )
    chunk_size = c2.slider("chunk_size (caracteres)", 100, 800, 300, step=50)
    chunk_overlap = c3.slider("chunk_overlap (caracteres)", 0, 400, 150, step=10)

    # Os sliders permitem overlap 400 e chunk_size 100, então a combinação
    # inválida é alcançável. Aqui NÃO se usa st.stop(): isso interromperia a
    # renderização da página inteira, inclusive o rodapé. Basta não montar a
    # seção de chunks.
    parametros_validos = chunk_overlap < chunk_size
    if not parametros_validos:
        st.error(
            f"O `chunk_overlap` ({chunk_overlap}) precisa ser **menor** que o "
            f"`chunk_size` ({chunk_size}). Reduza a sobreposição ou aumente o "
            f"tamanho do pedaço.",
            icon="⚠️",
        )

    if parametros_validos and texto_longo and texto_longo.strip():
        SEPARADORES = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

        if estrategia == "RecursiveCharacterTextSplitter":
            divisor = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                separators=SEPARADORES, length_function=len)
        else:
            divisor = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                separator=". ", length_function=len)

        chunks = divisor.split_text(texto_longo)

        n_tokens_total = len(modelo.tokenizer.encode(texto_longo,
                                                     add_special_tokens=True))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Caracteres", len(texto_longo))
        c2.metric("Tokens", n_tokens_total,
                  delta=f"{n_tokens_total - LIMITE_TOKENS:+d} vs limite",
                  delta_color="inverse")
        c3.metric("Chunks gerados", len(chunks))
        c4.metric("Tamanho médio",
                  f"{np.mean([len(c) for c in chunks]):.0f}" if chunks else "—")

        if n_tokens_total > LIMITE_TOKENS:
            perdido = (n_tokens_total - LIMITE_TOKENS) / n_tokens_total * 100
            st.warning(
                f"Sem chunking, **{perdido:.0f}% deste texto seria "
                f"descartado** pelo modelo.", icon="✂️")

        # ------------------------------------------------------------------
        # Overlap REAL. Esta verificação existe porque pedir overlap não
        # garante overlap: o splitter recursivo corta em separadores, e se o
        # orçamento de sobreposição for menor que uma frase, ele não consegue
        # repetir nada. Foi o que aconteceu na Entrega 3, com overlap 40 e 80.
        # ------------------------------------------------------------------
        def sobreposicao(a: str, b: str) -> int:
            """Maior trecho final de 'a' que é também o início de 'b'."""
            for n in range(min(len(a), len(b)), 0, -1):
                if a[-n:] == b[:n]:
                    return n
            return 0

        sobreposicoes = [sobreposicao(chunks[i], chunks[i + 1])
                         for i in range(len(chunks) - 1)]
        com_overlap = sum(1 for s in sobreposicoes if s > 0)

        st.markdown("#### O overlap pedido realmente aconteceu?")
        c1, c2 = st.columns(2)
        c1.metric("Pares consecutivos que compartilham texto",
                  f"{com_overlap} de {len(sobreposicoes)}" if sobreposicoes else "—")
        c2.metric("Média de caracteres sobrepostos",
                  f"{np.mean(sobreposicoes):.1f}" if sobreposicoes else "—")

        if sobreposicoes and com_overlap == 0 and chunk_overlap > 0:
            st.error(
                f"**Você pediu `chunk_overlap={chunk_overlap}`, mas nenhum "
                f"chunk compartilha texto com o seguinte.** O splitter corta "
                f"em separadores, não no meio da frase. Se o orçamento de "
                f"sobreposição for menor que uma frase deste texto, ele não "
                f"consegue repetir nada. Tente aumentar o `chunk_overlap`.",
                icon="🚨",
            )
        elif sobreposicoes and com_overlap > 0:
            st.success(
                f"Overlap efetivo em **{com_overlap} de {len(sobreposicoes)}** "
                f"transições, com média de {np.mean(sobreposicoes):.0f} "
                f"caracteres repetidos.", icon="✅")

        # --- Os chunks -----------------------------------------------------
        st.markdown("#### Chunks gerados")
        for i, c in enumerate(chunks, start=1):
            marca = ""
            if i > 1 and sobreposicoes[i - 2] > 0:
                marca = f" · repete {sobreposicoes[i - 2]} car. do anterior"
            with st.expander(f"Chunk {i} — {len(c)} caracteres{marca}",
                             expanded=(len(chunks) <= 4)):
                st.write(c)

        # --- Embeddings dos chunks ----------------------------------------
        if len(chunks) >= 2:
            st.divider()
            st.markdown("#### Embeddings dos chunks em 2D")

            emb_chunks = gerar_embeddings(id_modelo, tuple(chunks),
                                          papel="passage")

            # Com 3 ou mais chunks o PCA tem 2 componentes de verdade.
            # Com exatamente 2, só existe 1 direção de variação.
            if len(chunks) >= 3:
                coords_c = PCA(n_components=2,
                               random_state=42).fit_transform(emb_chunks)
            else:
                # Com 2 chunks o PCA só tem 1 componente útil
                coords_c = np.column_stack([
                    PCA(n_components=1, random_state=42).fit_transform(emb_chunks).ravel(),
                    np.zeros(len(chunks)),
                ])

            cdf = pd.DataFrame({
                "x": coords_c[:, 0], "y": coords_c[:, 1],
                "chunk": [f"{i}" for i in range(1, len(chunks) + 1)],
                "resumo": [c[:110] + "…" for c in chunks],
                "tamanho": [len(c) for c in chunks],
            })
            figc = px.scatter(cdf, x="x", y="y", text="chunk",
                              size="tamanho", size_max=28,
                              hover_data={"x": False, "y": False,
                                          "resumo": True, "tamanho": True},
                              color=cdf.index.astype(str),
                              color_discrete_sequence=px.colors.sequential.Blues_r)
            # Liga os chunks na ordem em que aparecem no texto
            figc.add_trace(go.Scatter(
                x=cdf["x"], y=cdf["y"], mode="lines",
                line=dict(color="#94a3b8", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False))
            figc.update_traces(textposition="middle center",
                               textfont=dict(color="white", size=11))
            figc.update_layout(height=440, showlegend=False,
                               margin=dict(l=10, r=10, t=30, b=10),
                               xaxis_title="PCA 1", yaxis_title="PCA 2")
            st.plotly_chart(figc, use_container_width=True)
            st.caption(
                "A linha pontilhada liga os chunks na ordem do texto. "
                "Saltos grandes indicam mudança de assunto — típico de "
                "manifestações que tratam de vários problemas de uma vez."
            )

            # --- Coesão entre chunks consecutivos -------------------------
            Sc = emb_chunks @ emb_chunks.T
            coesoes = [float(Sc[i, i + 1]) for i in range(len(chunks) - 1)]

            st.markdown("#### Coesão entre chunks consecutivos")
            st.caption(
                "Similaridade entre cada chunk e o seguinte. Valores altos "
                "indicam transição suave; valores baixos, mudança de assunto."
            )
            figl = go.Figure(go.Bar(
                x=[f"{i}→{i+1}" for i in range(1, len(chunks))],
                y=coesoes,
                marker_color=[faixa_de(c)[2] for c in coesoes],
                text=[f"{c:.3f}" for c in coesoes],
                textposition="auto",
            ))
            figl.add_hline(y=float(np.mean(coesoes)), line_dash="dash",
                           line_color="#64748b",
                           annotation_text=f"média {np.mean(coesoes):.3f}")
            figl.update_layout(height=300, showlegend=False,
                               margin=dict(l=10, r=10, t=30, b=10),
                               yaxis_title="similaridade",
                               yaxis_range=[0, 1])
            st.plotly_chart(figl, use_container_width=True)

            menor = int(np.argmin(coesoes))
            st.info(
                f"A transição mais brusca é **chunk {menor + 1} → "
                f"{menor + 2}** ({coesoes[menor]:.3f}). É provavelmente onde "
                f"a manifestação muda de assunto.", icon="🔎")
    elif parametros_validos:
        st.info("Escolha uma manifestação acima ou cole um texto.", icon="👆")

# ===========================================================================
# RODAPÉ
# ===========================================================================

st.divider()
st.caption(
    f"Ouvidoria Inteligente · Entrega 4 · corpus `{nome_corpus}` "
    f"({len(df)} manifestações) · modelo `{id_modelo.split('/')[-1]}` "
    f"({EMB.shape[1]} dimensões, limite de {LIMITE_TOKENS} tokens)"
)
