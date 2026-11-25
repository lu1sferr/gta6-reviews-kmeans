"""Análisis complementario de temas en las reseñas (TF-IDF + SVD + K-Means).

No forma parte del modelo principal del proyecto integrador (que segmenta con
variables numéricas). Se conserva como análisis exploratorio de texto.

Uso: python src/analisis_temas_texto.py
Salidas: data/processed/temas/*.csv y reports/figures/temas/*.png
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

SEED = 42
K_FINAL = 7
RAW, OUT, FIG = "data/raw/reviews.csv", "data/processed/temas", "reports/figures/temas"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---------- 1. Carga y limpieza ----------
df = pd.read_csv(RAW)
n0 = len(df)
df = df.drop_duplicates("review_id").drop_duplicates("review_text")
df["text"] = (df.review_text.str.replace(r"\\n", " ", regex=True)
              .str.replace(r"http\S+", "", regex=True).str.strip())
df["n_palabras"] = df.text.str.split().str.len()

STOP = {
    "en": set(ENGLISH_STOP_WORDS),
    "es": set("el la que pero juego muy es de y en un una los las por con para no se lo como mas más me mi esta está del al son ya".split()),
    "pt": set("jogo que um muito com melhor uma da os em não nao do é mas para por se mais".split()),
    "fr": set("est le jeu et les pas une des pour je la du un que il en ce sur".split()),
}


def idioma(t):
    w = re.findall(r"[a-záéíóúñãçêâ]+", t.lower())
    if not w:
        return "xx"
    sc = {l: sum(x in s for x in w) / len(w) for l, s in STOP.items()}
    b = max(sc, key=sc.get)
    return b if sc[b] > 0.08 else "xx"


df["idioma"] = df.text.map(idioma)

# Inglés con >= 8 palabras (mayoría y con texto suficiente)
en = df[(df.idioma == "en") & (df.n_palabras >= 8)].reset_index(drop=True)

# Review-bombing / ataques culturales: se separan antes de K-Means
en["review_bombing"] = en.text.str.contains(r"woke|cuck|\bsimp\b|feminis|slop|agenda", case=False)
print(f"Registros: {n0} -> {len(df)} sin duplicados -> {len(en)} en inglés útiles "
      f"({en.review_bombing.sum()} marcadas como posible review-bombing)")

en["sentimiento"] = pd.cut(en.score, [-1, 4, 6, 10], labels=["Negativa", "Mixta", "Positiva"])

# ---------- 2. Vectorización ----------
SW = list(ENGLISH_STOP_WORDS) + ["game", "gta", "don", "ve", "just", "im", "like", "really", "vi", "rockstar"]


def clusterizar(d, k):
    vec = TfidfVectorizer(stop_words=SW, ngram_range=(1, 2), min_df=10, max_df=0.4,
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(d.text)
    Z = Normalizer().fit_transform(TruncatedSVD(80, random_state=SEED).fit_transform(X))
    return vec, X, Z


# ---------- 3. K-Means sobre reseñas críticas ----------
crit = en[(en.score <= 6) & ~en.review_bombing].reset_index(drop=True)
vec, X, Z = clusterizar(crit, K_FINAL)
terms = np.array(vec.get_feature_names_out())

rows = []
for k in range(4, 11):
    km = KMeans(k, random_state=SEED, n_init=10).fit(Z)
    rows.append((k, km.inertia_, silhouette_score(Z, km.labels_, sample_size=6000, random_state=1)))
ev = pd.DataFrame(rows, columns=["K", "inercia", "silueta"])
ev.to_csv(f"{OUT}/evaluacion_k.csv", index=False)
print(ev.round(4).to_string(index=False))

km = KMeans(K_FINAL, random_state=SEED, n_init=10)
crit["cluster"] = km.fit_predict(Z)
km_b = KMeans(K_FINAL, random_state=SEED, n_init=3).fit(X)
sil_b = silhouette_score(X, km_b.labels_, sample_size=4000, random_state=1)
sil_a = silhouette_score(Z, crit.cluster, sample_size=6000, random_state=1)
print(f"Config A (SVD 80d) silueta={sil_a:.4f} | Config B (TF-IDF disperso) silueta={sil_b:.4f}")

info = []
for c in range(K_FINAL):
    m = (crit.cluster == c).values
    cen = np.asarray(X[m].mean(axis=0)).ravel()
    info.append(dict(cluster=c, n=int(m.sum()), score_prom=round(crit.score[m].mean(), 2),
                     terminos=", ".join(terms[cen.argsort()[::-1][:12]]),
                     ejemplo=crit.text[m].iloc[0][:220]))
info = pd.DataFrame(info)
info.to_csv(f"{OUT}/clusters_resumen.csv", index=False)
print(info[["cluster", "n", "score_prom", "terminos"]].to_string(index=False))

# ---------- 4. Etiquetado de aspectos ----------
ASPECTOS = {
    "Rendimiento (FPS/30fps)": r"\bfps\b|frame ?rate|30 ?fps|performance|stutter|lag",
    "Bugs / glitches / crashes": r"\bbugs?\b|glitch|crash|broken|freez",
    "Versión PC": r"\bpc\b",
    "Historia / ritmo": r"story|plot|pacing|boring|slow",
    "Personajes (Jason/Lucia)": r"jason|lucia|characters?",
    "Tareas/chores y misiones": r"chore|mission|errand|grind|repetitive",
    "Mundo abierto / libertad": r"open world|freedom|sandbox|alive",
    "Gráficos": r"graphics|visuals|visual",
    "Innovación vs GTA V": r"gta ?5|gta ?v\b|same|innovation|nothing new|recycl",
    "Precio / monetización": r"\bprice|\$|microtransaction|shark ?card|refund|overpriced|\bpaid\b|\bpay\b",
    "Online / multijugador": r"online|multiplayer|gta ?online",
    "Sobrecalentamiento": r"overheat|\bhot\b|temperature|\bfans?\b|melt|burn",
}
for a, rx in ASPECTOS.items():
    en[a] = en.text.str.contains(rx, case=False, regex=True)
asp = pd.DataFrame({
    "aspecto": list(ASPECTOS),
    "menciones": [int(en[a].sum()) for a in ASPECTOS],
    "score_prom": [round(en.score[en[a]].mean(), 2) for a in ASPECTOS],
    "pct_negativas": [round((en.score[en[a]] <= 4).mean() * 100, 1) for a in ASPECTOS],
}).sort_values("menciones", ascending=False)
asp.to_csv(f"{OUT}/aspectos.csv", index=False)
print(asp.to_string(index=False))

# ---------- 5. Salidas ----------
en = en.merge(crit[["review_id", "cluster"]], on="review_id", how="left")
en["cluster"] = en.cluster.fillna(-1).astype(int)
en.loc[en.review_bombing, "cluster"] = -2
en.drop(columns=["review_text", "text"]).to_csv(f"{OUT}/reviews_procesadas.csv", index=False)

plt.figure(figsize=(7, 4)); plt.plot(ev.K, ev.inercia, "o-"); plt.xlabel("K"); plt.ylabel("Inercia")
plt.title("Método del codo"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{FIG}/codo.png", dpi=150); plt.close()
plt.figure(figsize=(7, 4)); plt.plot(ev.K, ev.silueta, "o-", c="tab:green"); plt.xlabel("K"); plt.ylabel("Silueta")
plt.title("Coeficiente de silueta"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{FIG}/silueta.png", dpi=150); plt.close()

idx = np.random.RandomState(SEED).choice(len(crit), 4000, replace=False)
T = TSNE(2, random_state=SEED, init="pca", perplexity=40).fit_transform(Z[idx])
crit.loc[idx, "tsne_x"], crit.loc[idx, "tsne_y"] = T[:, 0], T[:, 1]
crit.loc[idx, ["review_id", "cluster", "tsne_x", "tsne_y"]].to_csv(f"{OUT}/tsne_muestra.csv", index=False)
plt.figure(figsize=(8, 6)); plt.scatter(T[:, 0], T[:, 1], c=crit.cluster.values[idx], cmap="tab10", s=6, alpha=.7)
plt.title("Clústeres de reseñas críticas (t-SNE)"); plt.tight_layout(); plt.savefig(f"{FIG}/tsne.png", dpi=150); plt.close()
print("Listo.")
