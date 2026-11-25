# GTA VI — Segmentación de reseñas con K-Means y dashboard interactivo

Proyecto integrador de minería de datos: segmentación no supervisada de **49.597 reseñas**
de usuarios de GTA VI (Metacritic, PlayStation 5 y Xbox Series X) mediante **K-Means**, con
un dashboard interactivo en Streamlit y un informe PDF reproducible.

El objetivo es identificar **perfiles de opinión** (no solo sentimiento) para priorizar mejoras
del juego y orientar la comunicación con la comunidad.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-KMeans-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## Resultados principales

Modelo final: **K = 5**, silueta **0,3324**, inercia **94.595,3**, 8 iteraciones.
Variables: puntaje, log(palabras), aspectos mencionados, negatividad léxica e intensidad
en mayúsculas, estandarizadas con `StandardScaler`.

| C | Perfil identificado | Reseñas | % | Puntaje medio | Palabras |
|---|---|---:|---:|---:|---:|
| C2 | Elogio breve | 17.061 | 34,4 % | 9,69 | 41 |
| C0 | Rechazo directo | 11.102 | 22,4 % | 1,04 | 41 |
| C1 | Análisis detallado y crítico | 10.337 | 20,8 % | 4,64 | 207 |
| C4 | Crítica negativa argumentada | 9.724 | 19,6 % | 1,90 | 40 |
| C3 | Protesta intensiva (mayúsculas) | 1.373 | 2,8 % | 4,26 | 40 |

Hallazgos: el promedio (5,02) oculta una distribución bipolar; el 42,0 % de las reseñas
pertenece a los dos grupos de rechazo; la crítica detallada concentra la información útil
para priorizar parches; historia/ritmo es el aspecto más mencionado (20.467 reseñas).
El detalle completo está en `docs/informe.pdf`.

---

## Estructura del repositorio

```
proyecto/
├── data/
│   ├── raw/reviews.csv                  # Dataset original (51.385 reseñas, 5 variables)
│   └── processed/                       # Tablas generadas por el pipeline
│       ├── resenas_segmentadas.csv      # Dataset procesado con clúster y coordenadas PCA
│       ├── resumen_clusters.csv         # Tabla comparativa de los 5 clústeres
│       ├── centroides_reales.csv        # Centroides en escala original
│       ├── centroides_estandarizados.csv
│       ├── distancias_centroides.csv    # Distancias de cada reseña a cada centroide
│       ├── distancias_muestra.csv       # Muestra de 1.000 casos para el dashboard
│       ├── evaluacion_k.csv             # Inercia y silueta para K = 2..10
│       ├── comparacion_configuraciones.csv
│       ├── estadisticas_clusters.csv    # Descriptivas por clúster (formato largo)
│       ├── perfil_aspectos.csv          # % de menciones de cada aspecto por clúster
│       ├── metricas.json                # Métricas usadas por dashboard e informe
│       └── temas/                       # Análisis complementario de texto (opcional)
├── dashboard/app.py                     # Dashboard Streamlit (2 páginas, 3 filtros)
├── docs/informe.pdf                     # Documento del proyecto (8 apartados)
├── reports/
│   ├── figures/                         # Gráficas del informe y del dashboard
│   └── dashboard/                       # Capturas de pantalla del dashboard
├── src/
│   ├── analisis_reviews.py              # Pipeline principal: EDA + limpieza + K-Means
│   ├── analisis_temas_texto.py          # Análisis complementario TF-IDF + SVD (opcional)
│   ├── capturas_dashboard.py            # Capturas automáticas con Playwright
│   └── generar_informe.py               # Genera docs/informe.pdf con ReportLab
├── requirements.txt
└── README.md
```

---

## Herramientas

- **Python 3.10+**, Pandas, NumPy
- **Scikit-learn**: `StandardScaler`, `KMeans`, `PCA`, `silhouette_score`, `adjusted_rand_score`
- **Matplotlib** y **Seaborn** para las gráficas del informe
- **Plotly** y **Streamlit** para el dashboard
- **ReportLab** y **Playwright** para generar el PDF y sus capturas

---

## Cómo ejecutar

```bash
# 1. Entorno virtual e instalación
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Pipeline completo (limpieza, K-Means, métricas y figuras)
python src/analisis_reviews.py

# 3. Dashboard
streamlit run dashboard/app.py
# Abrir http://localhost:8501
```

### Regenerar el informe PDF

El PDF incluye capturas del dashboard, por lo que conviene generarlas con el dashboard en
ejecución:

```bash
# Terminal 1
streamlit run dashboard/app.py

# Terminal 2
python -m playwright install chromium   # solo la primera vez
python src/capturas_dashboard.py
python src/generar_informe.py
```

Salida: `docs/informe.pdf`. Antes de entregar, edita los campos **Integrantes** y **Grupo**
en `src/generar_informe.py` (sección de portada) y vuelve a ejecutar el último comando.

### Análisis complementario de texto (opcional)

```bash
python src/analisis_temas_texto.py
```

Genera `data/processed/temas/` y `reports/figures/temas/` con un agrupamiento TF-IDF + SVD
sobre reseñas en inglés. No forma parte del modelo principal del proyecto.

---

## Dashboard

Dos páginas con tres filtros globales (**plataforma**, **rango de puntaje** y **clúster**):

**Página 1 — Análisis general**
- 4 KPI: reseñas analizadas, puntaje promedio, % de reseñas negativas y palabras promedio.
- Distribución de puntajes por plataforma y evolución diaria.
- Participación por plataforma y ranking de aspectos mencionados.
- Tabla de estadísticas descriptivas y primeros 10 registros.

**Página 2 — Segmentación K-Means**
- Métricas del modelo: K, silueta, inercia e iteraciones.
- Distribución de reseñas por clúster y dispersión PCA con centroides en rojo.
- Boxplot del puntaje, mapa de calor de aspectos por clúster.
- Tabla de estadísticas por clúster y tabla de distancias a los centroides.
- Descripción de cada grupo, hallazgos y recomendaciones.

Capturas: `reports/dashboard/dashboard_pagina1.png` y `dashboard_pagina2.png`.

---

## Metodología

1. **Limpieza**: 51.385 → 50.124 registros tras eliminar duplicados por `review_id` y texto;
   → 49.597 tras quitar reseñas con menos de 8 palabras. Sin valores nulos; puntajes válidos
   entre 0 y 10.
2. **Ingeniería de variables**: cinco variables numéricas por reseña (puntaje, log de palabras,
   cantidad de aspectos mencionados, negatividad léxica e intensidad en mayúsculas).
3. **Tratamiento de atípicos**: transformación logarítmica de la longitud y winsorización al
   percentil 99 de las proporciones (484 y 502 casos ajustados).
4. **Escalamiento**: `StandardScaler` (media 0, desviación 1).
5. **K-Means**: evaluación de K = 2..10 con inercia y silueta; K = 5 maximiza la silueta
   (0,3324) y coincide con el codo. `random_state=42`, `n_init=10`.
6. **Comparación**: configuración A (5 variables + escalado) frente a B (2 variables) y C
   (sin escalar). A se elige por equilibrio e interpretabilidad; la silueta de B no es
   directamente comparable por usar un espacio de menor dimensión.
7. **Interpretación**: centroides en escala real, estadísticas por grupo, perfiles de aspectos,
   nombres descriptivos y distancias euclidianas.

---

## Dataset

- **Fuente**: reseñas públicas de usuarios de Metacritic para GTA VI (PS5 y Xbox Series X).
- **Registros originales**: 51.385 · **columnas**: 5 · **nulos**: 0.
- **Periodo**: 19 al 24 de noviembre de 2026.
- **Variables originales**: `platform`, `review_id`, `published_date`, `score` (0–10),
  `review_text`.
- El archivo original se conserva sin modificaciones en `data/raw/reviews.csv`; el dataset con
  clústeres asignados está en `data/processed/resenas_segmentadas.csv`.

Uso académico. Los textos pertenecen a sus autores y a Metacritic; no se redistribuyen en el
dataset procesado (solo se conservan variables derivadas y metadatos).

---

## Limitaciones

- La silueta (0,3324) indica grupos con traslape, esperable en datos de opinión.
- La detección de aspectos usa palabras clave (inglés, español, portugués y francés) y puede
  no captar ironía, sinónimos o críticas implícitas.
- El análisis es observacional: los grupos describen asociaciones, no causalidad.
- Ventana de seis días y fuerte desbalance hacia PlayStation 5 (91,2 %).
- K-Means tiende a grupos esféricos y es sensible al escalamiento y a la selección de variables.

---

## Documento

`docs/informe.pdf` contiene los ocho apartados solicitados: portada, introducción y
planteamiento, objetivos, descripción del dataset, desarrollo, resultados y análisis,
dashboard y conclusiones/recomendaciones.
