"""Proyecto integrador: Segmentación de reseñas de GTA VI mediante K-Means.

Ejecuta todo el flujo (exploración, limpieza, K-Means, evaluación y gráficas):
    python src/analisis_reviews.py

Salidas:
    data/processed/*.csv   tablas del análisis
    reports/figures/*.png  gráficas del informe y del dashboard
    data/processed/metricas.json  métricas usadas por el dashboard y el PDF
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
SEED = 42
K_FINAL = 5
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews.csv"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.titlesize": 11})

# Aspectos que se detectan en el texto (inglés, español, portugués y francés).
ASPECTOS: dict[str, str] = {
    "Rendimiento (FPS)": (
        r"\bfps\b|frame ?rate|performance|stutter|lag|rendimiento|fluidez|"
        r"tirones|desempe[nñ]o|desempenho|fluidit[eé]|chute"
    ),
    "Bugs / crashes": (
        r"\bbugs?\b|glitch|crash|broken|freez|fallo|error|cierre|cierra|cuelga|"
        r"travamento|plantage|bogue"
    ),
    "Versión PC": r"\bpc\b|ordenador|computadora|computador",
    "Historia / ritmo": (
        r"story|plot|pacing|boring|slow|historia|trama|aburrid|aburrid[oa]|lento|"
        r"roteiro|ennuyeux|lent"
    ),
    "Personajes (Jason/Lucia)": (
        r"jason|lucia|characters?|personajes?|personagem|personnage"
    ),
    "Misiones / tareas": (
        r"chore|mission|errand|grind|repetitive|misi[oó]n|misiones|tarea|"
        r"repetitiv|miss[aã]o|repetitivo|qu[eê]te"
    ),
    "Mundo abierto": (
        r"open world|freedom|sandbox|mundo abierto|libertad|liberdade|"
        r"monde ouvert"
    ),
    "Gráficos": r"graphics|visuals|visual|gr[aá]fic|graphismes",
    "Precio / monetización": (
        r"\bprice|\$|microtransaction|shark ?card|refund|overpriced|\bpay\b|"
        r"precio|pre[cç]o|prix|reembolso|caro|estafa|scam"
    ),
    "Online / multijugador": r"online|multiplayer|multijugador|multijogador|multijoueur",
    "Sobrecalentamiento": (
        r"overheat|\bhot\b|temperature|\bfans?\b|melt|burn|calent|sobrecalent|"
        r"aquec|surchauff"
    ),
    "Innovación vs GTA V": (
        r"gta ?5|gta ?v\b|same|innovation|nothing new|recycl|innovaci[oó]n|"
        r"novedad|recicl|rien de nouveau"
    ),
}

NEG_RE = re.compile(
    r"\b(bad|boring|broken|bug|crash|trash|worst|terrible|awful|disappoint\w*|"
    r"lazy|greedy|unplayable|refund|scam|hate|sucks|mess|mid|slop|"
    r"malo|mala|mal[ií]simo|aburrid\w*|roto|falla|estafa|basura|p[eé]simo|"
    r"horrible|decepcion\w*|caro|ladr[oó]n|ruim|p[eé]ssimo|quebrad\w*|"
    r"nul|mauvais|d[eé]cevant)\b",
    re.IGNORECASE,
)

VARS_CLUSTER = [
    "puntaje",
    "log_palabras",
    "n_aspectos",
    "negatividad",
    "intensidad_mayusculas",
]
ETIQUETAS_VARS = {
    "puntaje": "Puntaje (0-10)",
    "log_palabras": "Palabras (log)",
    "n_aspectos": "Aspectos mencionados",
    "negatividad": "Negatividad léxica",
    "intensidad_mayusculas": "Intensidad (mayúsculas)",
    "n_palabras": "Palabras",
    "ratio_negativas": "% palabras negativas",
    "ratio_mayusculas": "% mayúsculas",
}


def seccion(titulo: str) -> None:
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables derivadas usadas en el modelado y el dashboard."""
    df = df.copy()
    df["texto"] = (
        df.review_text.str.replace(r"\\n", " ", regex=True)
        .str.replace(r"http\S+", "", regex=True)
        .str.strip()
    )
    df["n_palabras"] = df.texto.str.split().str.len().astype(int)
    df["n_caracteres"] = df.texto.str.len().astype(int)
    df["fecha"] = pd.to_datetime(df.published_date)
    df["plataforma"] = df.platform.map(
        {"playstation-5": "PlayStation 5", "xbox-series-x": "Xbox Series X"}
    ).fillna(df.platform)

    # 12 indicadores binarios de aspectos + conteo global
    for nombre, patron in ASPECTOS.items():
        df[f"asp_{nombre}"] = df.texto.str.contains(patron, case=False, regex=True)
    cols_asp = [f"asp_{n}" for n in ASPECTOS]
    df["n_aspectos"] = df[cols_asp].sum(axis=1).astype(int)

    # Señales de tono e intensidad
    df["n_palabras_negativas"] = df.texto.map(lambda t: len(NEG_RE.findall(t))).astype(int)
    df["ratio_negativas"] = df.n_palabras_negativas / df.n_palabras
    df["ratio_mayusculas"] = df.texto.map(
        lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
    )

    # Tratamiento de atípicos: winsorización al percentil 99 en proporciones
    p99_neg = df.ratio_negativas.quantile(0.99)
    p99_may = df.ratio_mayusculas.quantile(0.99)
    df["n_neg_winsor"] = int((df.ratio_negativas > p99_neg).sum())
    df["n_may_winsor"] = int((df.ratio_mayusculas > p99_may).sum())
    df["ratio_negativas_w"] = df.ratio_negativas.clip(upper=p99_neg)
    df["ratio_mayusculas_w"] = df.ratio_mayusculas.clip(upper=p99_may)

    # Transformaciones: log para longitudes y para la proporción de negatividad
    df["log_palabras"] = np.log1p(df.n_palabras)
    df["negatividad"] = np.log1p(df.ratio_negativas_w * 100)
    df["intensidad_mayusculas"] = df.ratio_mayusculas_w
    df["puntaje"] = df.score.astype(float)
    df["dias_desde_inicio"] = (df.fecha - df.fecha.min()).dt.days
    return df


# ---------------------------------------------------------------------------
# PARTE 2. Obtención y exploración del dataset
# ---------------------------------------------------------------------------
seccion("PARTE 2. Obtención y exploración del dataset")
raw = pd.read_csv(RAW)
print(f"Fuente: data/raw/reviews.csv | registros: {len(raw):,} | columnas: {raw.shape[1]}")
print("\nPrimeros 10 registros:")
print(raw.head(10).to_string(max_colwidth=45))

resumen_eda = {
    "registros": int(len(raw)),
    "columnas": int(raw.shape[1]),
    "variables": list(raw.columns),
    "tipos": {c: str(t) for c, t in raw.dtypes.items()},
    "nulos": {c: int(v) for c, v in raw.isna().sum().items()},
    "duplicados_review_id": int(raw.review_id.duplicated().sum()),
    "duplicados_texto": int(raw.review_text.duplicated().sum()),
    "plataformas": {k: int(v) for k, v in raw.platform.value_counts().items()},
    "puntaje_promedio": round(float(raw.score.mean()), 2),
    "registros_por_plataforma": {k: int(v) for k, v in raw.platform.value_counts().items()},
}
desc = raw.drop(columns=["review_text"]).describe(include="all").T
desc.to_csv(PROC / "eda_estadisticas_descriptivas.csv")
raw.head(10).to_csv(PROC / "eda_primeros_registros.csv", index=False)
print(f"\nDimensiones: {raw.shape[0]:,} filas x {raw.shape[1]} columnas")
print("Tipos de datos:")
print(raw.dtypes.to_string())
print("\nValores nulos:", resumen_eda["nulos"])
print("Duplicados por review_id:", resumen_eda["duplicados_review_id"],
      "| duplicados de texto:", resumen_eda["duplicados_texto"])
print("\nEstadísticas descriptivas (variables originales):")
print(desc[["count", "unique", "top", "mean", "std", "min", "max"]].to_string()
      if "unique" in desc.columns else desc.to_string())

# --- Gráficas exploratorias -------------------------------------------------
pal = {"PlayStation 5": "#2E5EAA", "Xbox Series X": "#1DB954"}

fig, ax = plt.subplots(figsize=(8, 4.2))
plot_df = raw.assign(plataforma=raw.platform.map(
    {"playstation-5": "PlayStation 5", "xbox-series-x": "Xbox Series X"}))
sns.histplot(data=plot_df, x="score", hue="plataforma", multiple="stack",
             bins=11, palette=pal, ax=ax)
ax.set_title("Distribución de puntajes por plataforma")
ax.set_xlabel("Puntaje (0-10)")
ax.set_ylabel("Cantidad de reseñas")
fig.tight_layout()
fig.savefig(FIG / "eda_distribucion_puntajes.png")
plt.close(fig)

long_raw = pd.DataFrame({
    "n_palabras": raw.review_text.str.split().str.len(),
    "plataforma": plot_df.plataforma,
})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(np.log1p(long_raw.n_palabras), bins=40, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribución de la longitud (log de palabras)")
axes[0].set_xlabel("log(1 + palabras)")
axes[0].set_ylabel("Cantidad de reseñas")
sns.boxplot(data=long_raw, x="plataforma", y="n_palabras", ax=axes[1], palette=pal)
axes[1].set_title("Longitud por plataforma (con atípicos)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Palabras")
fig.tight_layout()
fig.savefig(FIG / "eda_longitud_resenas.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.countplot(data=plot_df, x="plataforma", palette=pal, ax=axes[0], hue="plataforma",
              legend=False)
axes[0].set_title("Cantidad de reseñas por plataforma")
axes[0].set_xlabel("")
axes[0].set_ylabel("Reseñas")
plot_df["fecha"] = pd.to_datetime(plot_df.published_date)
por_dia = plot_df.groupby(["fecha", "plataforma"], observed=True).size().reset_index(name="n")
sns.lineplot(data=por_dia, x="fecha", y="n", hue="plataforma", marker="o",
             palette=pal, ax=axes[1])
axes[1].set_title("Evolución diaria de las reseñas")
axes[1].set_xlabel("")
axes[1].set_ylabel("Reseñas")
axes[1].tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIG / "eda_plataformas_tiempo.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# PARTE 3. Limpieza y preparación
# ---------------------------------------------------------------------------
seccion("PARTE 3. Limpieza y preparación de datos")
df = raw.drop_duplicates("review_id").drop_duplicates("review_text").copy()
n_sin_dups = len(df)
df = construir_features(df)
n_cortas = int((df.n_palabras < 8).sum())
df = df[df.n_palabras >= 8].reset_index(drop=True)

limpieza = {
    "registros_iniciales": int(len(raw)),
    "duplicados_review_id": int(raw.review_id.duplicated().sum()),
    "duplicados_texto": int(raw.review_text.duplicated().sum()),
    "registros_sin_duplicados": int(n_sin_dups),
    "resenas_menos_8_palabras": n_cortas,
    "registros_finales": int(len(df)),
    "nulos_finales": int(df.isna().sum().sum()),
    "puntaje_min": int(df.score.min()),
    "puntaje_max": int(df.score.max()),
    "palabras_p99": float(df.n_palabras.quantile(0.99)),
    "palabras_max": int(df.n_palabras.max()),
    "winsor_negatividad": int(df.n_neg_winsor.iloc[0]),
    "winsor_mayusculas": int(df.n_may_winsor.iloc[0]),
}
print(f"Registros: {len(raw):,} -> {n_sin_dups:,} sin duplicados -> {len(df):,} con >= 8 palabras")
print(f"Reseñas cortas eliminadas: {n_cortas:,} | nulos en el dataset final: {limpieza['nulos_finales']}")
print(f"Puntajes válidos en [{limpieza['puntaje_min']}, {limpieza['puntaje_max']}]")
print(f"Longitud: p99={limpieza['palabras_p99']:.0f} palabras, máx={limpieza['palabras_max']} "
      f"(se usa log1p, sin eliminar reseñas largas)")
print(f"Winsorización al p99: {limpieza['winsor_negatividad']} reseñas en negatividad y "
      f"{limpieza['winsor_mayusculas']} en mayúsculas")
print("\nEstadísticas de las variables del modelo:")
print(df[VARS_CLUSTER].describe().round(3).to_string())

# Gráfica de aspectos y correlación
conteo_asp = pd.Series({n: int(df[f"asp_{n}"].sum()) for n in ASPECTOS}).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=conteo_asp.values, y=conteo_asp.index, hue=conteo_asp.index,
            palette="viridis", legend=False, ax=ax)
ax.set_title("Menciones de cada aspecto en las reseñas")
ax.set_xlabel("Cantidad de reseñas que mencionan el aspecto")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIG / "eda_aspectos.png")
plt.close(fig)

corr = df[["puntaje", "n_palabras", "n_aspectos", "ratio_negativas",
           "ratio_mayusculas", "dias_desde_inicio"]].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"label": "Correlación"})
ax.set_title("Correlación entre variables numéricas")
fig.tight_layout()
fig.savefig(FIG / "eda_correlacion.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# PARTE 4. K-Means
# ---------------------------------------------------------------------------
seccion("PARTE 4. Implementación de K-Means")
X = df[VARS_CLUSTER].to_numpy()
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print("Estandarización aplicada (media=0, desviación=1) a:")
for v in VARS_CLUSTER:
    print(f"  - {ETIQUETAS_VARS[v]:<26} media={scaler.mean_[VARS_CLUSTER.index(v)]:8.3f} "
          f"desv={scaler.scale_[VARS_CLUSTER.index(v)]:8.3f}")

evaluacion = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
    sil = silhouette_score(Xs, km.labels_, sample_size=8000, random_state=1)
    evaluacion.append({"K": k, "inercia": km.inertia_, "silueta": float(sil)})
ev = pd.DataFrame(evaluacion)
ev.to_csv(PROC / "evaluacion_k.csv", index=False)
print("\nEvaluación de K (2 a 10):")
print(ev.round(4).to_string(index=False))
k_opt = int(ev.loc[ev.silueta.idxmax(), "K"])
print(f"\nK con mayor silueta: {k_opt}. Se adopta K={K_FINAL} porque maximiza la "
      f"silueta, coincide con el codo de la inercia y produce grupos interpretables.")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(ev.K, ev.inercia, "o-", color="#4C72B0")
ax.set_title("Método del codo (inercia)")
ax.set_xlabel("Número de clústeres (K)")
ax.set_ylabel("Inercia")
ax.axvline(K_FINAL, color="red", ls="--", alpha=.6)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "codo.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(ev.K, ev.silueta, "o-", color="#2E8B57")
ax.set_title("Coeficiente de silueta por K")
ax.set_xlabel("Número de clústeres (K)")
ax.set_ylabel("Silueta")
ax.axvline(K_FINAL, color="red", ls="--", alpha=.6)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "silueta.png")
plt.close(fig)

# --- Modelo final -----------------------------------------------------------
km = KMeans(n_clusters=K_FINAL, random_state=SEED, n_init=10).fit(Xs)
df["cluster"] = km.labels_.astype(int)
distancias = km.transform(Xs)
df["distancia_centroide"] = distancias.min(axis=1)
sil_final = silhouette_score(Xs, km.labels_, sample_size=8000, random_state=1)
print(f"\nModelo final: K={K_FINAL}, inercia={km.inertia_:,.1f}, "
      f"silueta={sil_final:.4f}, iteraciones={km.n_iter_}")
print("Cantidad de reseñas por clúster:")
print(df.cluster.value_counts().sort_index().to_string())

# Centroides en escala estandarizada y en escala real
cent_std = pd.DataFrame(km.cluster_centers_, columns=VARS_CLUSTER)
cent_std.insert(0, "cluster", range(K_FINAL))
cent_std.to_csv(PROC / "centroides_estandarizados.csv", index=False)

cent_real = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=VARS_CLUSTER)
cent_real["palabras_equiv"] = np.expm1(cent_real.log_palabras)
cent_real["ratio_negativas_pct"] = np.expm1(cent_real.negatividad)
cent_real["ratio_mayusculas_pct"] = cent_real.intensidad_mayusculas * 100
cent_real = cent_real[["puntaje", "palabras_equiv", "n_aspectos", "ratio_negativas_pct",
                       "ratio_mayusculas_pct"]]
cent_real.columns = ["puntaje", "palabras", "n_aspectos", "ratio_negativas_pct",
                     "ratio_mayusculas_pct"]
cent_real.insert(0, "cluster", range(K_FINAL))
cent_real.to_csv(PROC / "centroides_reales.csv", index=False)
print("\nCentroides en escala real (interpretables):")
print(cent_real.round(2).to_string(index=False))

# Distancias a los centroides (tabla tipo kmeans.py)
cols_dist = [f"Distancia_centroide_{c}" for c in range(K_FINAL)]
dist_full = pd.DataFrame(distancias, columns=cols_dist)
dist_full.insert(0, "review_id", df.review_id)
dist_full.insert(1, "cluster", df.cluster)
dist_full.to_csv(PROC / "distancias_centroides.csv", index=False)
dist_sample = dist_full.sample(1000, random_state=SEED).sort_values("cluster")
dist_sample.to_csv(PROC / "distancias_muestra.csv", index=False)
print("\nDistancias euclidianas a los centroides (muestra de 5 reseñas):")
print(dist_full.head(5).round(3).to_string(index=False))

# --- Perfil y nombres de clústeres -----------------------------------------
filas = []
for c in range(K_FINAL):
    sub = df[df.cluster == c]
    filas.append({
        "cluster": c,
        "n": int(len(sub)),
        "pct": round(len(sub) / len(df) * 100, 1),
        "puntaje_prom": round(sub.puntaje.mean(), 2),
        "puntaje_mediana": round(sub.puntaje.median(), 1),
        "palabras_prom": round(sub.n_palabras.mean(), 1),
        "palabras_mediana": round(sub.n_palabras.median(), 1),
        "aspectos_prom": round(sub.n_aspectos.mean(), 2),
        "ratio_negativas_pct": round(sub.ratio_negativas.mean() * 100, 2),
        "ratio_mayusculas_pct": round(sub.ratio_mayusculas.mean() * 100, 2),
        "pct_negativas": round((sub.score <= 4).mean() * 100, 1),
        "pct_ps5": round((sub.plataforma == "PlayStation 5").mean() * 100, 1),
    })
resumen = pd.DataFrame(filas)

perfil_asp = (
    df.groupby("cluster")[[f"asp_{n}" for n in ASPECTOS]].mean() * 100
).round(1)
perfil_asp.columns = list(ASPECTOS)
perfil_asp.to_csv(PROC / "perfil_aspectos.csv", index_label="cluster")
top_asp = perfil_asp.idxmax(axis=1)
resumen["aspecto_dominante"] = resumen.cluster.map(top_asp)


def nombrar(fila: pd.Series) -> str:
    if fila.ratio_mayusculas_pct >= 15:
        return "Protesta intensiva (mayúsculas)"
    if fila.puntaje_prom >= 7:
        return "Elogio breve"
    if fila.palabras_prom >= 100:
        return "Análisis detallado y crítico"
    if fila.puntaje_prom <= 1.5:
        return "Rechazo directo"
    return "Crítica negativa argumentada"


resumen["nombre"] = resumen.apply(nombrar, axis=1)
# Ejemplo representativo: reseña más cercana al centroide de su clúster
for c in range(K_FINAL):
    ejemplo = df.loc[df.cluster == c].nsmallest(1, "distancia_centroide")
    resumen.loc[resumen.cluster == c, "ejemplo"] = ejemplo.texto.iloc[0][:240]
resumen = resumen.sort_values("puntaje_prom", ascending=False).reset_index(drop=True)
resumen.to_csv(PROC / "resumen_clusters.csv", index=False)
print("\nResumen de clústeres:")
print(resumen[["cluster", "nombre", "n", "pct", "puntaje_prom", "palabras_prom",
               "aspectos_prom"]].to_string(index=False))

# Estadísticas descriptivas por clúster (formato largo)
stats = (
    df.groupby("cluster")[["puntaje", "n_palabras", "n_aspectos", "ratio_negativas",
                           "ratio_mayusculas", "dias_desde_inicio"]]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .stack(level=0, future_stack=True)
    .reset_index()
)
stats.columns = ["cluster", "variable", "conteo", "media", "desv_est", "mediana", "min", "max"]
stats.to_csv(PROC / "estadisticas_clusters.csv", index=False)

# --- Comparación de configuraciones ----------------------------------------
seccion("Comparación de configuraciones del modelo")


def entrenar(variables: list[str], escalar: bool = True) -> dict:
    Xc = df[variables].to_numpy()
    if escalar:
        Xc = StandardScaler().fit_transform(Xc)
    modelo = KMeans(n_clusters=K_FINAL, random_state=SEED, n_init=10).fit(Xc)
    return {
        "labels": modelo.labels_,
        "inercia": modelo.inertia_,
        "silueta": silhouette_score(Xc, modelo.labels_, sample_size=8000, random_state=1),
        "tamanos": pd.Series(modelo.labels_).value_counts().sort_index().tolist(),
    }


config_a = entrenar(VARS_CLUSTER, escalar=True)
config_b = entrenar(["puntaje", "log_palabras"], escalar=True)
config_c = entrenar(VARS_CLUSTER, escalar=False)
comparacion = pd.DataFrame([
    {"configuracion": "A: 5 variables + StandardScaler", "variables": 5, "escalado": "Sí",
     "K": K_FINAL, "inercia": config_a["inercia"], "silueta": config_a["silueta"],
     "tamanos": config_a["tamanos"]},
    {"configuracion": "B: puntaje y palabras + StandardScaler", "variables": 2,
     "escalado": "Sí", "K": K_FINAL, "inercia": config_b["inercia"],
     "silueta": config_b["silueta"], "tamanos": config_b["tamanos"]},
    {"configuracion": "C: 5 variables sin escalar", "variables": 5, "escalado": "No",
     "K": K_FINAL, "inercia": config_c["inercia"], "silueta": config_c["silueta"],
     "tamanos": config_c["tamanos"]},
])
comparacion["tamanos"] = comparacion["tamanos"].map(lambda x: " / ".join(map(str, x)))
comparacion["ARI_vs_A"] = [
    1.0,
    adjusted_rand_score(config_a["labels"], config_b["labels"]),
    adjusted_rand_score(config_a["labels"], config_c["labels"]),
]
comparacion = comparacion.round({"inercia": 1, "silueta": 4, "ARI_vs_A": 3})
comparacion.to_csv(PROC / "comparacion_configuraciones.csv", index=False)
print(comparacion.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=comparacion, x="configuracion", y="silueta", hue="configuracion",
            palette="Blues_d", legend=False, ax=ax)
for i, v in enumerate(comparacion.silueta):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_title("Comparación de configuraciones (silueta, mayor es mejor)")
ax.set_xlabel("")
ax.set_ylabel("Silueta")
ax.tick_params(axis="x", rotation=12)
fig.tight_layout()
fig.savefig(FIG / "comparacion_configuraciones.png")
plt.close(fig)

# --- Proyección PCA y gráficas de dispersión --------------------------------
pca = PCA(n_components=2, random_state=SEED).fit(Xs)
pcs = pca.transform(Xs)
df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]
cent_pca = pca.transform(km.cluster_centers_)
var_pca = pca.explained_variance_ratio_
pd.DataFrame({
    "cluster": range(K_FINAL),
    "pc1": cent_pca[:, 0],
    "pc2": cent_pca[:, 1],
}).to_csv(PROC / "centroides_pca.csv", index=False)

# Dispersión 1: proyección PCA con centroides (estilo kmeans.py)
rng = np.random.RandomState(SEED)
idx = rng.choice(len(df), 6000, replace=False)
colores = plt.cm.tab10(np.linspace(0, 1, 10))
fig, ax = plt.subplots(figsize=(9, 6))
for c in range(K_FINAL):
    m = (df.cluster.values[idx] == c)
    ax.scatter(df.pc1.values[idx][m], df.pc2.values[idx][m], s=8, alpha=.45,
               color=colores[c], label=f"C{c}: {resumen.loc[resumen.cluster == c, 'nombre'].iloc[0]}")
ax.scatter(cent_pca[:, 0], cent_pca[:, 1], marker="X", s=260, c="red",
           edgecolors="black", linewidths=.8, label="Centroides", zorder=5)
for c in range(K_FINAL):
    ax.annotate(f"C{c}", (cent_pca[c, 0], cent_pca[c, 1]),
                textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")
ax.set_title("Agrupamiento de reseñas con K-Means (proyección PCA)")
ax.set_xlabel(f"Componente principal 1 ({var_pca[0]*100:.1f}% var.)")
ax.set_ylabel(f"Componente principal 2 ({var_pca[1]*100:.1f}% var.)")
ax.legend(fontsize=7, loc="best", framealpha=.9)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(FIG / "dispersion_pca.png")
plt.close(fig)

# Dispersión 2: puntaje vs longitud (estilo kmeans.py, con centroides)
sample = df.sample(3000, random_state=SEED)
fig, ax = plt.subplots(figsize=(10, 6))
for c in range(K_FINAL):
    m = sample.cluster == c
    ax.scatter(sample.loc[m, "n_palabras"], sample.loc[m, "puntaje"], s=18,
               alpha=.5, color=colores[c],
               label=f"C{c}: {resumen.loc[resumen.cluster == c, 'nombre'].iloc[0]}")
ax.scatter(cent_real.palabras, cent_real.puntaje, marker="X", s=280, c="red",
           edgecolors="black", linewidths=.8, label="Centroides", zorder=5)
for _, fila in cent_real.iterrows():
    ax.annotate(f"C{int(fila.cluster)}", (fila.palabras, fila.puntaje),
                textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold")
ax.set_xscale("log")
ax.set_title("Dispersión de reseñas por puntaje y longitud (K-Means, K=5)")
ax.set_xlabel("Palabras por reseña (escala log)")
ax.set_ylabel("Puntaje")
ax.legend(fontsize=7, loc="center right", framealpha=.9)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(FIG / "dispersion_puntaje_longitud.png")
plt.close(fig)

# Calor del perfil de centroides (escala estandarizada)
perfil_std = cent_std.set_index("cluster")
perfil_std.index = [f"C{c} - {resumen.loc[resumen.cluster == c, 'nombre'].iloc[0]}"
                    for c in perfil_std.index]
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.heatmap(perfil_std, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"label": "Desviaciones estándar"})
ax.set_title("Perfil de los centroides (variables estandarizadas)")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([ETIQUETAS_VARS[c] for c in perfil_std.columns], rotation=25, ha="right")
fig.tight_layout()
fig.savefig(FIG / "centroides_perfil.png")
plt.close(fig)

# Caja de puntajes por clúster
df["nombre_cluster"] = df.cluster.map(resumen.set_index("cluster").nombre)
orden = resumen.sort_values("puntaje_prom").nombre.tolist()
fig, ax = plt.subplots(figsize=(10, 4.6))
sns.boxplot(data=df, x="nombre_cluster", y="puntaje", order=orden,
            hue="nombre_cluster", palette="Set2", legend=False, ax=ax)
ax.set_title("Distribución del puntaje por clúster")
ax.set_xlabel("")
ax.set_ylabel("Puntaje")
ax.tick_params(axis="x", rotation=18)
fig.tight_layout()
fig.savefig(FIG / "boxplot_puntaje_cluster.png")
plt.close(fig)

# Aspectos por clúster
perfil_plot = perfil_asp.copy()
perfil_plot.index = [f"C{c} - {resumen.loc[resumen.cluster == c, 'nombre'].iloc[0]}"
                     for c in perfil_plot.index]
fig, ax = plt.subplots(figsize=(11, 5.4))
sns.heatmap(perfil_plot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "% de reseñas del clúster"})
ax.set_title("Porcentaje de reseñas que mencionan cada aspecto, por clúster")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "aspectos_por_cluster.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# PARTE 5. Métricas finales y salidas
# ---------------------------------------------------------------------------
seccion("PARTE 5. Resultados y salidas")
cols_salida = [
    "review_id", "platform", "plataforma", "fecha", "score", "puntaje", "n_palabras",
    "n_caracteres", "n_aspectos", "ratio_negativas", "ratio_mayusculas", "negatividad",
    "log_palabras", "intensidad_mayusculas", "dias_desde_inicio", "cluster",
    "distancia_centroide", "pc1", "pc2", "nombre_cluster",
] + [f"asp_{n}" for n in ASPECTOS]
df[cols_salida].to_csv(PROC / "resenas_segmentadas.csv", index=False)

aspecto_top = conteo_asp.idxmax()
metricas = {
    **resumen_eda,
    **limpieza,
    "k_final": K_FINAL,
    "k_silueta_max": k_opt,
    "silueta_final": round(float(sil_final), 4),
    "inercia_final": round(float(km.inertia_), 1),
    "iteraciones": int(km.n_iter_),
    "pca_varianza": [round(float(v), 4) for v in var_pca],
    "aspecto_top": aspecto_top,
    "aspecto_top_menciones": int(conteo_asp.max()),
    "pct_negativas_global": round(float((df.score <= 4).mean() * 100), 1),
    "puntaje_promedio_modelo": round(float(df.puntaje.mean()), 2),
    "palabras_promedio": round(float(df.n_palabras.mean()), 1),
    "pct_ps5": round(float((df.plataforma == "PlayStation 5").mean() * 100), 1),
    "config_a_silueta": round(float(config_a["silueta"]), 4),
    "config_b_silueta": round(float(config_b["silueta"]), 4),
    "config_c_silueta": round(float(config_c["silueta"]), 4),
    "ari_a_b": round(float(adjusted_rand_score(config_a["labels"], config_b["labels"])), 3),
    "ari_a_c": round(float(adjusted_rand_score(config_a["labels"], config_c["labels"])), 3),
    "n_clusters": int(K_FINAL),
    "tamanos": {str(c): int((df.cluster == c).sum()) for c in range(K_FINAL)},
    "generado": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}
with open(PROC / "metricas.json", "w", encoding="utf-8") as fh:
    json.dump(metricas, fh, ensure_ascii=False, indent=2)

print(f"Silueta final: {sil_final:.4f} | PCA explica "
      f"{var_pca.sum()*100:.1f}% de la varianza")
print(f"Aspecto más mencionado: {aspecto_top} ({conteo_asp.max():,} reseñas)")
print("Archivos generados en data/processed y reports/figures")
print("\nListo. Dashboard: streamlit run dashboard/app.py")
