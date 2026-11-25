"""Dashboard interactivo del proyecto de segmentación de reseñas de GTA VI.

Ejecutar desde la raíz del proyecto:
    streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="GTA VI: voz del jugador", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stSidebar"] {min-width: 390px; width: 390px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cargar():
    d = pd.read_csv(PROC / "resenas_segmentadas.csv", parse_dates=["fecha"])
    res = pd.read_csv(PROC / "resumen_clusters.csv")
    perfil = pd.read_csv(PROC / "perfil_aspectos.csv")
    ev = pd.read_csv(PROC / "evaluacion_k.csv")
    cent_pca = pd.read_csv(PROC / "centroides_pca.csv")
    dist = pd.read_csv(PROC / "distancias_muestra.csv")
    with open(PROC / "metricas.json", encoding="utf-8") as fh:
        met = json.load(fh)
    return d, res, perfil, ev, cent_pca, dist, met


d, res, perfil, ev, cent_pca, dist, met = cargar()
res = res.sort_values("cluster").reset_index(drop=True)
nombre_por_cluster = dict(zip(res.cluster, res.nombre))
asp_cols = [c for c in d.columns if c.startswith("asp_")]
asp_labels = [c.replace("asp_", "") for c in asp_cols]

# ---------------------------------------------------------------------------
# Filtros (3 filtros interactivos)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Navegación")
    qp = st.query_params.get("pagina", "1")
    pagina = st.radio(
        "Página",
        ["1. Análisis general", "2. Segmentación K-Means"],
        index=1 if str(qp) == "2" else 0,
    )
    st.divider()
    st.header("Filtros")
    plataformas = sorted(d.plataforma.unique())
    sel_plat = st.multiselect("Plataforma", plataformas, plataformas)
    rango_score = st.slider("Rango de puntaje", 0, 10, (0, 10))
    nombres = res.nombre.tolist()
    sel_grupo = st.multiselect("Clúster", nombres, nombres)
    st.caption("Los filtros afectan a los indicadores y a las gráficas de las dos páginas.")

f = d[
    d.plataforma.isin(sel_plat)
    & d.puntaje.between(*rango_score)
    & d.nombre_cluster.isin(sel_grupo)
].copy()

st.title("GTA VI — Segmentación de reseñas con K-Means")
st.caption(
    f"Fuente: Metacritic (PS5 y Xbox Series X) · {met['registros']:,} reseñas originales · "
    f"{met['registros_finales']:,} analizadas · modelo K={met['k_final']} · "
    f"silueta={met['silueta_final']:.3f}"
)

if f.empty:
    st.warning("No hay reseñas con los filtros seleccionados. Ajusta la plataforma, el puntaje o el clúster.")
    st.stop()

# ---------------------------------------------------------------------------
# Página 1. Análisis general
# ---------------------------------------------------------------------------
if pagina.startswith("1"):
    c = st.columns(4)
    c[0].metric("Reseñas analizadas", f"{len(f):,}")
    c[1].metric("Puntaje promedio", f"{f.puntaje.mean():.2f}")
    c[2].metric("% negativas (≤4)", f"{(f.puntaje <= 4).mean() * 100:.1f}%")
    c[3].metric("Palabras promedio", f"{f.n_palabras.mean():.0f}")

    a, b = st.columns(2)
    hist = px.histogram(
        f, x="puntaje", color="plataforma", nbins=11, barmode="group",
        title="Distribución de puntajes por plataforma",
        labels={"puntaje": "Puntaje (0-10)", "count": "Reseñas", "plataforma": "Plataforma"},
        color_discrete_sequence=["#2E5EAA", "#1DB954"],
    )
    hist.update_layout(bargap=0.08, legend_title_text="")
    a.plotly_chart(hist, width="stretch")

    por_dia = f.groupby([f.fecha.dt.date, "plataforma"], observed=True).size().reset_index(name="reseñas")
    por_dia.columns = ["fecha", "plataforma", "reseñas"]
    linea = px.line(
        por_dia, x="fecha", y="reseñas", color="plataforma", markers=True,
        title="Evolución diaria de las reseñas",
        color_discrete_sequence=["#2E5EAA", "#1DB954"],
    )
    linea.update_layout(legend_title_text="")
    b.plotly_chart(linea, width="stretch")

    a, b = st.columns([1, 1.6])
    plat = f.plataforma.value_counts().reset_index()
    plat.columns = ["plataforma", "reseñas"]
    pastel = px.pie(
        plat, names="plataforma", values="reseñas", hole=0.45,
        title="Participación por plataforma",
        color="plataforma", color_discrete_sequence=["#2E5EAA", "#1DB954"],
    )
    a.plotly_chart(pastel, width="stretch")

    conteo = f[asp_cols].sum().sort_values()
    conteo.index = [i.replace("asp_", "") for i in conteo.index]
    barras = px.bar(
        x=conteo.values, y=conteo.index, orientation="h",
        title="Aspectos más mencionados en las reseñas",
        labels={"x": "Reseñas que mencionan el aspecto", "y": ""},
        color=conteo.values, color_continuous_scale="Teal",
    )
    barras.update_layout(coloraxis_showscale=False)
    b.plotly_chart(barras, width="stretch")

    st.subheader("Estadísticas descriptivas generales")
    tabla = pd.DataFrame({
        "Indicador": [
            "Reseñas", "Puntaje promedio", "Puntaje mediana", "Desviación estándar",
            "Palabras promedio", "Palabras mediana", "Aspectos promedio",
            "% reseñas negativas (≤4)", "% reseñas positivas (≥8)",
        ],
        "Valor": [
            f"{len(f):,}", f"{f.puntaje.mean():.2f}", f"{f.puntaje.median():.1f}",
            f"{f.puntaje.std():.2f}", f"{f.n_palabras.mean():.0f}",
            f"{f.n_palabras.median():.0f}", f"{f.n_aspectos.mean():.2f}",
            f"{(f.puntaje <= 4).mean() * 100:.1f}%",
            f"{(f.puntaje >= 8).mean() * 100:.1f}%",
        ],
    })
    st.dataframe(tabla, hide_index=True, width="stretch")

    with st.expander("Ver los primeros 10 registros del dataset"):
        cols = ["review_id", "plataforma", "fecha", "puntaje", "n_palabras", "n_aspectos",
                "nombre_cluster"]
        st.dataframe(f[cols].head(10), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Página 2. Segmentación mediante K-Means
# ---------------------------------------------------------------------------
else:
    c = st.columns(4)
    c[0].metric("Clústeres (K)", met["k_final"])
    c[1].metric("Coeficiente de silueta", f"{met['silueta_final']:.3f}")
    c[2].metric("Inercia final", f"{met['inercia_final']:,.0f}")
    c[3].metric("Iteraciones", met["iteraciones"])

    a, b = st.columns([1, 1.4])
    tam = f.nombre_cluster.value_counts().reset_index()
    tam.columns = ["clúster", "reseñas"]
    dist_bar = px.bar(
        tam, x="reseñas", y="clúster", orientation="h",
        title="Distribución de reseñas por clúster",
        color="clúster", text="reseñas",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    dist_bar.update_layout(showlegend=False, yaxis_title="")
    a.plotly_chart(dist_bar, width="stretch")

    cp = cent_pca.merge(res[["cluster", "nombre"]], on="cluster")
    cp = cp[cp.nombre.isin(sel_grupo)]
    fig = px.scatter(
        f, x="pc1", y="pc2", color="nombre_cluster", opacity=0.45,
        title="Dispersión de los agrupamientos (proyección PCA) con centroides",
        labels={"pc1": f"PC1 ({met['pca_varianza'][0] * 100:.1f}% var.)",
                "pc2": f"PC2 ({met['pca_varianza'][1] * 100:.1f}% var.)",
                "nombre_cluster": "Clúster"},
        render_mode="webgl",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_trace(go.Scatter(
        x=cp.pc1, y=cp.pc2, mode="markers+text",
        marker=dict(symbol="x", size=18, color="red", line=dict(width=2, color="black")),
        text=[f"C{int(c)}" for c in cp.cluster],
        textposition="top center",
        textfont=dict(size=12, color="black"),
        name="Centroides",
        hovertemplate="Centroide C%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
    ))
    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
    b.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    box = px.box(
        f, x="nombre_cluster", y="puntaje", color="nombre_cluster",
        title="Puntaje por clúster",
        labels={"nombre_cluster": "", "puntaje": "Puntaje"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    box.update_layout(showlegend=False, xaxis_tickangle=-15)
    a.plotly_chart(box, width="stretch")

    perfil_f = (f.groupby("nombre_cluster")[asp_cols].mean() * 100).round(1)
    perfil_f.columns = asp_labels
    perfil_f = perfil_f.loc[[n for n in res.nombre if n in perfil_f.index]]
    calor = px.imshow(
        perfil_f, aspect="auto", color_continuous_scale="YlOrRd", text_auto=".0f",
        title="% de reseñas de cada clúster que mencionan el aspecto",
        labels={"x": "", "y": "", "color": "% del clúster"},
    )
    calor.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    b.plotly_chart(calor, width="stretch")

    st.subheader("Estadísticas descriptivas por clúster")
    stats = f.groupby("nombre_cluster").agg(
        Reseñas=("review_id", "count"),
        Puntaje_medio=("puntaje", "mean"),
        Puntaje_mediana=("puntaje", "median"),
        Palabras_medio=("n_palabras", "mean"),
        Aspectos_medio=("n_aspectos", "mean"),
        Pct_negativas=("puntaje", lambda s: (s <= 4).mean() * 100),
        Pct_mayusculas=("ratio_mayusculas", lambda s: s.mean() * 100),
    )
    stats["% del total"] = stats.Reseñas / len(f) * 100
    stats = stats.loc[[n for n in res.nombre if n in stats.index]]
    stats.columns = ["Reseñas", "Puntaje medio", "Puntaje mediana", "Palabras medio",
                     "Aspectos medio", "% negativas", "% mayúsculas", "% del total"]
    st.dataframe(
        stats.style.format({
            "Puntaje medio": "{:.2f}", "Puntaje mediana": "{:.1f}",
            "Palabras medio": "{:.0f}", "Aspectos medio": "{:.2f}",
            "% negativas": "{:.1f}%", "% mayúsculas": "{:.1f}%", "% del total": "{:.1f}%",
        }),
        width="stretch",
    )

    with st.expander("Tabla de distancias de cada reseña a los centroides (muestra)"):
        dm = dist.copy()
        dm["clúster"] = dm.cluster.map(nombre_por_cluster)
        dm = dm[["review_id", "clúster"] + [c for c in dm.columns if c.startswith("Distancia")]]
        st.dataframe(dm.head(200), hide_index=True, width="stretch")
        st.caption("Distancias euclidianas en el espacio de las 5 variables estandarizadas. "
                   "El clúster asignado es el del centroide más cercano.")

    st.subheader("Descripción de los grupos identificados")
    for _, fila in res.iterrows():
        if fila.nombre not in sel_grupo:
            continue
        with st.container(border=True):
            st.markdown(
                f"**C{int(fila.cluster)} · {fila.nombre}** — {fila.n:,} reseñas ({fila.pct:.1f}%)"
            )
            st.markdown(
                f"- **Puntaje medio:** {fila.puntaje_prom:.2f} · "
                f"**Palabras medias:** {fila.palabras_prom:.0f} · "
                f"**Aspectos medios:** {fila.aspectos_prom:.2f}\n"
                f"- **% de reseñas negativas (≤4):** {fila.pct_negativas:.1f}% · "
                f"**% de mayúsculas:** {fila.ratio_mayusculas_pct:.1f}%\n"
                f"- **Aspecto más mencionado:** {fila.aspecto_dominante}"
            )
            st.caption(f"Ejemplo representativo: “{fila.ejemplo}…”")

    st.subheader("Principales hallazgos y recomendaciones")
    mayor = res.loc[res.n.max() == res.n, "nombre"].iloc[0]
    negativo = res[res.puntaje_prom < 3]
    pct_neg_clusters = negativo.n.sum() / res.n.sum() * 100
    detallado = res.loc[res.nombre == "Análisis detallado y crítico"]
    neg_por_aspecto = (d.groupby("asp_Precio / monetización").puntaje.apply(
        lambda s: (s <= 4).mean() * 100))
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("#### Hallazgos")
        st.markdown(
            f"1. **El grupo más grande es “{mayor}”** con {res.loc[res.nombre == mayor, 'pct'].iloc[0]:.1f}% "
            f"de las reseñas; el tono dominante es {'positivo' if 'Elogio' in mayor else 'crítico'}.\n"
            f"2. **{pct_neg_clusters:.1f}% de las reseñas pertenece a los dos grupos de rechazo** "
            f"(puntaje medio menor a 3), lo que concentra la insatisfacción.\n"
            f"3. **La crítica detallada** ({detallado.pct.iloc[0]:.1f}% del total) escribe en promedio "
            f"{detallado.palabras_prom.iloc[0]:.0f} palabras y menciona "
            f"{detallado.aspectos_prom.iloc[0]:.1f} aspectos: es el segmento con información más útil "
            f"para priorizar mejoras.\n"
            f"4. **{met['aspecto_top']}** es el aspecto más mencionado "
            f"({met['aspecto_top_menciones']:,} reseñas); en precio/monetización, "
            f"{neg_por_aspecto.get(True, 0):.1f}% de quienes lo mencionan puntúan ≤4."
        )
    with h2:
        st.markdown("#### Recomendaciones")
        st.markdown(
            "1. **Priorizar narrativa y ritmo:** es el aspecto más discutido; conviene revisar el "
            "pacing de misiones y la escritura de los protagonistas.\n"
            "2. **Plan de calidad técnica:** atender bugs, caídas de FPS y versión PC, que aparecen "
            "con fuerza en el clúster de análisis detallado.\n"
            "3. **Comunicación segmentada:** usar las reseñas detalladas como fuente de backlog y "
            "responder al rechazo directo con mensajes breves y concretos.\n"
            "4. **Monitorear la protesta intensiva:** el clúster de mayúsculas actúa como señal "
            "temprana de malestar y permite anticipar picos de review-bombing."
        )

    st.caption(
        "Limitaciones: la silueta (0.33) indica grupos con traslape, esperable en datos de opinión; "
        "la detección de aspectos se basa en palabras clave y puede no captar ironías ni sinónimos. "
        "Los resultados describen asociaciones, no relaciones causales."
    )
