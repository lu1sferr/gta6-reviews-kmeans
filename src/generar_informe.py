"""Genera el informe PDF del proyecto integrador.

Uso: python src/generar_informe.py
Salida: docs/informe.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
DASH = ROOT / "reports" / "dashboard"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Fuentes y estilos
# ---------------------------------------------------------------------------
FUENTES = Path("/usr/share/fonts/TTF")
pdfmetrics.registerFont(TTFont("DejaVu", str(FUENTES / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FUENTES / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FUENTES / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(FUENTES / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

AZUL = colors.HexColor("#173B63")
AZUL_CLARO = colors.HexColor("#E8EEF6")
GRIS = colors.HexColor("#555555")

S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=25, leading=31,
                            alignment=TA_CENTER, textColor=AZUL),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=13, leading=18,
                               alignment=TA_CENTER, textColor=GRIS),
    "kicker": ParagraphStyle("kicker", fontName="DejaVu-Bold", fontSize=9, leading=12,
                             alignment=TA_CENTER, textColor=colors.HexColor("#7A8CA3")),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15.5, leading=20,
                         textColor=AZUL, spaceBefore=4, spaceAfter=8, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15,
                         textColor=AZUL, spaceBefore=10, spaceAfter=5, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.8, leading=14,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.8, leading=14,
                             alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4,
                             spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Oblique", fontSize=8.2, leading=11,
                              alignment=TA_CENTER, textColor=GRIS, spaceBefore=2,
                              spaceAfter=10),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8.5, leading=11.5),
    "cover": ParagraphStyle("cover", fontName="DejaVu", fontSize=10.5, leading=16),
}

met = json.loads((PROC / "metricas.json").read_text(encoding="utf-8"))
res = pd.read_csv(PROC / "resumen_clusters.csv").sort_values("cluster").reset_index(drop=True)
cent = pd.read_csv(PROC / "centroides_reales.csv").sort_values("cluster")
ev = pd.read_csv(PROC / "evaluacion_k.csv")
comp = pd.read_csv(PROC / "comparacion_configuraciones.csv")
perfil = pd.read_csv(PROC / "perfil_aspectos.csv").set_index("cluster")

N = met["registros_finales"]
nombres = dict(zip(res.cluster, res.nombre))
c_elogio = res.set_index("nombre").loc["Elogio breve"]
c_detalle = res.set_index("nombre").loc["Análisis detallado y crítico"]
c_rechazo = res.set_index("nombre").loc["Rechazo directo"]
c_critica = res.set_index("nombre").loc["Crítica negativa argumentada"]
c_protesta = res.set_index("nombre").loc["Protesta intensiva (mayúsculas)"]
pct_rechazo = c_rechazo.pct + c_critica.pct

# Recortes de la captura de la página 2 para que quepa en el PDF
pag2 = DASH / "dashboard_pagina2.png"
if pag2.exists():
    im = PILImage.open(pag2)
    ancho_img, alto_img = im.size
    im.crop((0, 0, ancho_img, 1600)).save(DASH / "dashboard_pagina2_a.png")
    im.crop((390, 1600, ancho_img, min(3480, alto_img))).save(DASH / "dashboard_pagina2_b.png")


def p(texto: str, estilo: str = "body") -> Paragraph:
    return Paragraph(texto, S[estilo])


def vineta(texto: str) -> Paragraph:
    return Paragraph(texto, S["bullet"], bulletText="•")


def imagen(ruta: Path, ancho_cm: float = 15.0, alto_max_cm: float = 19.5,
           titulo: str | None = None) -> KeepTogether:
    from PIL import Image as _Im
    iw, ih = _Im.open(ruta).size
    alto_cm = ancho_cm * ih / iw
    if alto_cm > alto_max_cm:
        alto_cm = alto_max_cm
        ancho_cm = alto_cm * iw / ih
    bloques = [Image(str(ruta), width=ancho_cm * cm, height=alto_cm * cm)]
    if titulo:
        bloques.append(p(titulo, "caption"))
    return KeepTogether(bloques)


def _celda(valor, negrita: bool, fs: float, alinear: str, color=colors.black) -> Paragraph:
    estilo = ParagraphStyle(
        "celda", fontName="DejaVu-Bold" if negrita else "DejaVu", fontSize=fs,
        leading=fs * 1.22, textColor=color,
        alignment=TA_LEFT if alinear == "LEFT" else TA_CENTER,
    )
    return Paragraph(str(valor), estilo)


def tabla(datos, anchos=None, fs=8.2, alinear="CENTER") -> Table:
    filas = [
        [_celda(v, i == 0, fs, alinear, colors.white if i == 0 else colors.black) for v in fila]
        for i, fila in enumerate(datos)
    ]
    t = Table(filas, colWidths=anchos, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, AZUL_CLARO]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def pie_pagina(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DCE5"))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.45 * cm, A4[0] - 2 * cm, 1.45 * cm)
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(GRIS)
    canvas.drawString(2 * cm, 1.0 * cm, "GTA VI — Segmentación de reseñas con K-Means")
    canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(DOCS / "informe.pdf"), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
    title="Segmentación de reseñas de GTA VI mediante K-Means",
    author="Proyecto integrador de minería de datos",
)
H = []  # historia del documento

# ---------------------------------------------------------------------------
# 1. PORTADA
# ---------------------------------------------------------------------------
H += [Spacer(1, 2.6 * cm), p("PROYECTO INTEGRADOR · MINERÍA DE DATOS", "kicker"),
      Spacer(1, 0.35 * cm)]
H.append(Table([[""]], colWidths=[6 * cm], rowHeights=[0.12 * cm],
               style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), AZUL)])))
H += [Spacer(1, 1.0 * cm),
      p("Segmentación de reseñas de GTA VI mediante K-Means y dashboard interactivo", "title"),
      Spacer(1, 0.6 * cm),
      p("Identificación de perfiles de jugadores a partir de 49.597 reseñas de Metacritic", "subtitle"),
      Spacer(1, 2.2 * cm)]

portada = [
    [p("<b>Integrantes</b>", "cover"), p("____________________________________________", "cover")],
    [p("<b>Grupo</b>", "cover"), p("____________________________________________", "cover")],
    [p("<b>Fecha</b>", "cover"), p("24 de noviembre de 2026", "cover")],
    [p("<b>Herramientas</b>", "cover"),
     p("Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit", "cover")],
]
tp = Table(portada, colWidths=[3.4 * cm, 10.6 * cm], hAlign="CENTER")
tp.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE5")),
]))
H += [tp, Spacer(1, 2.4 * cm),
      p("Análisis no supervisado de opiniones de usuarios de PlayStation 5 y Xbox Series X "
        "publicadas durante la primera semana de lanzamiento.", "subtitle"),
      PageBreak()]

# ---------------------------------------------------------------------------
# 2. INTRODUCCIÓN Y PLANTEAMIENTO DEL PROBLEMA
# ---------------------------------------------------------------------------
H.append(p("2. Introducción y planteamiento del problema", "h1"))
H.append(p(
    "El lanzamiento de <b>Grand Theft Auto VI</b> el 19 de noviembre de 2026 generó un volumen "
    "extraordinario de opiniones en muy pocos días. En la fuente utilizada se reunieron "
    f"<b>{met['registros']:,} reseñas</b> de usuarios de Metacritic para PlayStation 5 y Xbox "
    "Series X entre el 19 y el 24 de noviembre de 2026. Leer y sintetizar manualmente esa "
    "cantidad de texto es inviable: una sola persona tardaría semanas y perdería de vista los "
    "patrones que solo se observan al analizar el conjunto completo."))
H.append(p(
    "El problema central es que <b>el promedio oculta la estructura real de la opinión</b>. "
    f"El puntaje promedio del conjunto es {met['puntaje_promedio_modelo']:.2f} sobre 10, un valor "
    "aparentemente neutro que, sin embargo, esconde una distribución bimodal: "
    f"{met['pct_negativas_global']:.1f}% de las reseñas otorga 4 puntos o menos y 41,8% otorga "
    "8 puntos o más. Es decir, conviven un rechazo intenso y un elogio entusiasta, con muy pocos "
    "matices intermedios. Tratar ambos extremos como una sola opinión promedio conduce a "
    "decisiones equivocadas."))
H.append(p(
    "Para abordar esta situación se propone <b>segmentar las reseñas con K-Means</b>, un "
    "algoritmo de aprendizaje no supervisado que agrupa registros con características similares. "
    "A diferencia de un análisis de sentimiento tradicional, el agrupamiento permite distinguir "
    "no solo si la opinión es positiva o negativa, sino <b>cómo opina cada jugador</b>: si "
    "escribe mucho o poco, si menciona varios aspectos del juego, si utiliza un tono "
    "emocionalmente intenso o si fundamenta su crítica con argumentos técnicos."))
H.append(p("Para ello se construyeron cinco variables numéricas a partir de cada reseña: "
           "el puntaje otorgado, la longitud del texto, la cantidad de aspectos del juego "
           "mencionados, la proporción de palabras negativas y la intensidad expresada mediante "
           "el uso de mayúsculas. Estas variables describen el comportamiento y el estilo de la "
           "opinión, no solo su polaridad."))
H.append(p(
    "<b>Justificación.</b> El proyecto aporta valor en tres frentes. En primer lugar, permite a "
    "un equipo de desarrollo o de publicación <b>priorizar mejoras</b> a partir de los aspectos "
    "que aparecen con mayor frecuencia y peor valoración. En segundo lugar, separa la "
    "<b>crítica argumentada</b> de la <b>reacción breve</b>, dos tipos de opinión que requieren "
    "respuestas distintas. En tercer lugar, identifica un segmento de <b>protesta intensiva</b> "
    "que puede actuar como señal temprana de malestar y de dinámicas de review-bombing. Todo "
    "ello se presenta en un dashboard interactivo que permite explorar los grupos por "
    "plataforma, puntaje y clúster."))
H.append(p(
    "<b>Pregunta de investigación.</b> ¿Qué perfiles de opinión pueden identificarse entre las "
    "reseñas de GTA VI mediante K-Means, y cómo puede esa segmentación orientar la priorización "
    "de mejoras y la comunicación con la comunidad?"))
H.append(p(
    "El alcance se limita a las reseñas publicadas en las dos plataformas disponibles durante "
    "la primera semana de lanzamiento. El análisis es descriptivo y exploratorio: los grupos "
    "representan asociaciones entre variables, no relaciones causales."))

# ---------------------------------------------------------------------------
# 3. OBJETIVOS
# ---------------------------------------------------------------------------
H.append(p("3. Objetivos", "h1"))
H.append(p("<b>Objetivo general</b>", "h2"))
H.append(p(
    "Segmentar las reseñas de GTA VI mediante el algoritmo K-Means para identificar perfiles "
    "de opinión, interpretar sus características y comunicar los resultados en un dashboard "
    "interactivo que facilite la toma de decisiones."))
H.append(p("<b>Objetivos específicos</b>", "h2"))
for texto in [
    "Limpiar y preparar el dataset, construyendo al menos cinco variables numéricas que "
    "describan el puntaje, la extensión, el enfoque temático, el tono y la intensidad de cada reseña.",
    "Evaluar al menos ocho valores de K mediante la inercia y el coeficiente de silueta, y "
    "seleccionar y justificar el número de clústeres más adecuado.",
    "Interpretar los centroides y las estadísticas por grupo, asignar un nombre descriptivo a "
    "cada clúster y comparar dos configuraciones del modelo.",
    "Diseñar un dashboard interactivo de dos páginas con indicadores, filtros, visualizaciones "
    "y una sección de hallazgos y recomendaciones derivada de los datos.",
]:
    H.append(vineta(texto))

# ---------------------------------------------------------------------------
# 4. DESCRIPCIÓN DEL DATASET
# ---------------------------------------------------------------------------
H.append(p("4. Descripción del dataset", "h1"))
H.append(p(
    "La fuente es el conjunto público de <b>reseñas de usuarios de Metacritic</b> para GTA VI, "
    "correspondiente a las versiones de PlayStation 5 y Xbox Series X. Los datos se almacenan en "
    "el archivo <font face='DejaVu-Bold'>data/raw/reviews.csv</font>. El conjunto original "
    f"contiene <b>{met['registros']:,} registros</b> y <b>{met['columnas']} variables</b>, "
    "publicados entre el 19 y el 24 de noviembre de 2026. No se encontraron valores nulos en "
    "ninguna columna."))
H.append(p("<b>Variables originales</b>", "h2"))
datos_vars = [
    ["Variable", "Tipo", "Descripción"],
    ["platform", "Texto", "Plataforma de la reseña: playstation-5 o xbox-series-x"],
    ["review_id", "Texto", "Identificador único de la reseña"],
    ["published_date", "Fecha", "Fecha de publicación (formato AAAA-MM-DD)"],
    ["score", "Entero", "Puntaje otorgado por el usuario, de 0 a 10"],
    ["review_text", "Texto", "Contenido completo de la opinión"],
]
H.append(tabla(datos_vars, anchos=[3.0 * cm, 2.0 * cm, 11.6 * cm], fs=8.4, alinear="LEFT"))
H.append(p("Tabla 1. Variables originales del dataset.", "caption"))
H.append(Spacer(1, 0.15 * cm))
H.append(p(
    f"La distribución por plataforma está desbalanceada: {met['pct_ps5']:.1f}% de los registros "
    "finales proviene de PlayStation 5 y el resto de Xbox Series X. Este desbalance se conserva "
    "y se hace explícito en el dashboard mediante un filtro, en lugar de eliminar registros. "
    "La longitud de las reseñas es muy variable: la mediana es de 39 palabras, la media de 75 y "
    "el máximo de 1.364 palabras, lo que justifica una transformación logarítmica."))
H.append(p("<b>Exploración inicial</b>", "h2"))
H.append(p(
    "Se revisaron dimensiones, tipos de datos, estadísticas descriptivas, valores nulos y "
    "duplicados. El dataset original no presenta nulos, pero sí 1.156 identificadores "
    "duplicados y 651 textos repetidos. La distribución de puntajes es marcadamente bimodal "
    "(Figura 1), con acumulación en 0 y en 10. La actividad se concentra entre el 20 y el 22 de "
    "noviembre (Figura 2). Los aspectos más mencionados son la historia y el ritmo, seguidos por "
    "los personajes, los gráficos y los bugs."))
H.append(imagen(FIG / "eda_distribucion_puntajes.png", 13.5,
                titulo="Figura 1. Distribución de puntajes por plataforma (dataset original)."))
H.append(imagen(FIG / "eda_longitud_resenas.png", 15,
                titulo="Figura 2. Longitud de las reseñas y comparación por plataforma."))
H.append(imagen(FIG / "eda_aspectos.png", 13,
                titulo="Figura 3. Cantidad de reseñas que mencionan cada aspecto."))
H.append(p("<b>Variables seleccionadas para el agrupamiento</b>", "h2"))
H.append(p(
    "Se construyeron cinco variables numéricas. La selección busca capturar dimensiones "
    "complementarias de la opinión y evitar que el modelo se limite a reproducir el puntaje:"))
datos_feats = [
    ["Variable", "Construcción", "Justificación"],
    ["puntaje", "score (0 a 10)", "Satisfacción declarada; eje principal de la opinión."],
    ["log_palabras", "log(1 + palabras)", "Extensión y nivel de elaboración; reduce la asimetría."],
    ["n_aspectos", "Suma de 12 indicadores temáticos", "Amplitud de la crítica: cuántos temas aborda."],
    ["negatividad", "log(1 + 100 · palabras negativas / palabras)", "Tono léxico independiente del puntaje."],
    ["intensidad_mayusculas", "% de mayúsculas (winsorizado al p99)", "Intensidad emocional o énfasis."],
]
H.append(tabla(datos_feats, anchos=[3.9 * cm, 5.3 * cm, 7.3 * cm], fs=7.6, alinear="LEFT"))
H.append(p("Tabla 2. Variables construidas para el agrupamiento.", "caption"))
H.append(Spacer(1, 0.1 * cm))
H.append(p(
    "Los 12 aspectos detectados son: rendimiento (FPS), bugs y cierres, versión PC, historia y "
    "ritmo, personajes, misiones y tareas, mundo abierto, gráficos, precio y monetización, "
    "online, sobrecalentamiento e innovación frente a GTA V. La detección se realiza con "
    "expresiones regulares en inglés, español, portugués y francés, por lo que cada variable "
    "cuenta cuántos de esos temas aparecen en el texto."))

# ---------------------------------------------------------------------------
# 5. DESARROLLO DEL PROYECTO
# ---------------------------------------------------------------------------
H.append(p("5. Desarrollo del proyecto", "h1"))
H.append(p("<b>Limpieza y preparación</b>", "h2"))
H.append(p(
    "El flujo completo se implementó en Python con Pandas y Scikit-learn "
    "(<font face='DejaVu-Bold'>src/analisis_reviews.py</font>). La limpieza siguió estos pasos: "
    "eliminación de duplicados por identificador y por texto; eliminación de reseñas con menos "
    "de ocho palabras, que no aportan suficiente contenido; verificación de nulos y de tipos; "
    "conversión de la fecha a tipo datetime; y validación del rango del puntaje entre 0 y 10. "
    "La Tabla 3 resume el efecto de cada etapa."))
datos_limpieza = [
    ["Etapa", "Registros", "Observación"],
    ["Dataset original", f"{met['registros']:,}", "5 variables, sin valores nulos"],
    ["Duplicados por review_id", f"-{met['duplicados_review_id']:,}", "Se conserva el primero"],
    ["Duplicados por texto", f"-{met['duplicados_texto']:,}", "Coincidencia exacta del texto"],
    ["Sin duplicados", f"{met['registros_sin_duplicados']:,}", "Base depurada"],
    ["Reseñas con menos de 8 palabras", f"-{met['resenas_menos_8_palabras']:,}", "Texto insuficiente"],
    ["Dataset final", f"{met['registros_finales']:,}", "Usado en el modelado"],
]
H.append(tabla(datos_limpieza, anchos=[5.6 * cm, 3.0 * cm, 8.0 * cm], fs=8.4))
H.append(p("Tabla 3. Resultado de la limpieza y preparación.", "caption"))
H.append(Spacer(1, 0.2 * cm))
H.append(p(
    "Los valores atípicos se trataron de forma diferenciada. La longitud de las reseñas presenta "
    f"un máximo de {met['palabras_max']:,} palabras y un percentil 99 de {met['palabras_p99']:.0f}; "
    "en lugar de eliminar reseñas largas, se aplicó una transformación logarítmica que reduce la "
    "influencia de los extremos sin perder información. Las proporciones de palabras negativas y "
    "de mayúsculas se winsorizaron en el percentil 99 "
    f"({met['winsor_negatividad']} y {met['winsor_mayusculas']} casos ajustados, respectivamente) "
    "para evitar que unos pocos textos extremos dominaran los centroides. Finalmente, las cinco "
    "variables se estandarizaron con <b>StandardScaler</b> (media 0, desviación 1), requisito "
    "esencial en K-Means porque el algoritmo usa distancias euclidianas y las escalas originales "
    "son incomparables."))
H.append(p(
    "La estandarización es importante por dos razones. Primero, impide que una variable con "
    "mayor recorrido, como la longitud en palabras, opaque a las demás. Segundo, hace que los "
    "centroides sean comparables entre sí y que las distancias tengan un significado homogéneo. "
    "Sin escalamiento, el puntaje y la longitud dominarían la formación de los grupos."))
H.append(p("<b>Implementación de K-Means</b>", "h2"))
H.append(p(
    "Se utilizó <font face='DejaVu-Bold'>KMeans</font> de Scikit-learn con "
    "random_state=42 y n_init=10 para estabilizar la inicialización. Se evaluaron nueve valores "
    "de K, de 2 a 10, calculando la inercia y el coeficiente de silueta sobre una muestra de "
    "8.000 registros para controlar el costo computacional. La Tabla 4 y las Figuras 4 y 5 "
    "muestran los resultados."))
filas_k = [["K", "Inercia", "Silueta"]]
for _, r in ev.iterrows():
    filas_k.append([str(int(r.K)), f"{r.inercia:,.1f}", f"{r.silueta:.4f}"])
H.append(tabla(filas_k, anchos=[2.2 * cm, 4.0 * cm, 3.2 * cm], fs=8.6))
H.append(p("Tabla 4. Evaluación de K mediante inercia y silueta.", "caption"))
H.append(Spacer(1, 0.15 * cm))
H.append(imagen(FIG / "codo.png", 9.2, alto_max_cm=8.5, titulo="Figura 4. Método del codo."))
H.append(imagen(FIG / "silueta.png", 9.2, alto_max_cm=8.5,
                titulo="Figura 5. Coeficiente de silueta para cada valor de K."))
H.append(p(
    f"El mayor coeficiente de silueta se obtiene con <b>K = {met['k_final']}</b> "
    f"({met['silueta_final']:.4f}), valor que además coincide con el cambio de pendiente de la "
    "inercia. Se adoptó K = 5 porque maximiza la métrica, coincide con el codo y produce grupos "
    "interpretables y con tamaños suficientes. El modelo final alcanzó una inercia de "
    f"{met['inercia_final']:,.1f} y convergió en {met['iteraciones']} iteraciones."))
H.append(p("<b>Centroides y distancias</b>", "h2"))
H.append(p(
    "Cada centroide representa el perfil promedio de un grupo en el espacio de las cinco "
    "variables. Para interpretarlos se invirtió la estandarización y se recuperó la escala "
    "original (Tabla 5). La distancia euclidiana de cada reseña a cada centroide se calculó con "
    "<font face='DejaVu-Bold'>transform()</font>; la reseña se asigna al centroide más cercano. "
    "La tabla completa se guarda en data/processed/distancias_centroides.csv y una muestra de "
    "1.000 casos se presenta en el dashboard."))
filas_cent = [["Clúster", "Puntaje", "Palabras", "Aspectos", "% negativas", "% mayúsculas"]]
for _, r in cent.iterrows():
    filas_cent.append([f"C{int(r.cluster)}", f"{r.puntaje:.2f}", f"{r.palabras:.0f}",
                       f"{r.n_aspectos:.2f}", f"{r.ratio_negativas_pct:.2f}%",
                       f"{r.ratio_mayusculas_pct:.1f}%"])
H.append(tabla(filas_cent, anchos=[2.2 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 3.4 * cm, 3.4 * cm],
               fs=8.4))
H.append(p("Tabla 5. Centroides en escala original (promedios del perfil de cada grupo).", "caption"))
H.append(p("<b>Comparación de configuraciones</b>", "h2"))
H.append(p(
    "Se compararon tres configuraciones con el mismo K = 5: (A) las cinco variables con "
    "StandardScaler, (B) solo puntaje y longitud con escalamiento, y (C) las cinco variables sin "
    "escalar. La Tabla 6 reporta inercia, silueta y el índice de Rand ajustado (ARI) respecto de "
    "la configuración A."))
filas_comp = [["Configuración", "Variables", "Escalado", "Silueta", "ARI vs A", "Tamaños"]]
for _, r in comp.iterrows():
    filas_comp.append([r.configuracion.replace(" + StandardScaler", ""), str(r.variables),
                       str(r.escalado), f"{r.silueta:.4f}", f"{r.ARI_vs_A:.3f}", r.tamanos])
H.append(tabla(filas_comp, anchos=[4.2 * cm, 1.7 * cm, 1.6 * cm, 1.7 * cm, 1.9 * cm, 5.5 * cm],
               fs=7.4))
H.append(p("Tabla 6. Comparación de configuraciones del modelo.", "caption"))
H.append(Spacer(1, 0.15 * cm))
H.append(p(
    "La configuración B obtiene la silueta más alta (0,4808), pero se calcula en un espacio de "
    "solo dos dimensiones y, por tanto, <b>no es directamente comparable</b> con la de cinco "
    "dimensiones: al reducir el número de variables, los puntos quedan artificialmente más "
    "compactos. La configuración C también obtiene una silueta alta (0,4458), pero sin escalar "
    "el puntaje y la longitud dominan la distancia y los grupos pierden matices. La "
    "configuración A presenta la silueta más baja (0,3324), aunque es la única que utiliza las "
    "cinco variables de forma equilibrada y produce los cinco perfiles interpretables que se "
    "analizan en la sección 6. El ARI de 0,404 entre A y B confirma que B reorganiza buena parte "
    "de los registros: pierde la distinción entre rechazo directo y crítica argumentada. Por "
    "estas razones se adopta A como modelo final."))
H.append(imagen(FIG / "comparacion_configuraciones.png", 12,
                titulo="Figura 6. Coeficiente de silueta de las tres configuraciones."))

# ---------------------------------------------------------------------------
# 6. RESULTADOS Y ANÁLISIS
# ---------------------------------------------------------------------------
H.append(p("6. Resultados y análisis", "h1"))
H.append(p(
    f"El modelo final segmentó las <b>{N:,} reseñas</b> en cinco grupos. La silueta de "
    f"{met['silueta_final']:.4f} indica una separación moderada: los grupos son temas con "
    "traslape, algo esperable en datos de opinión, donde las fronteras no son nítidas. La "
    "proyección PCA utilizada para visualizar los grupos explica "
    f"{sum(met['pca_varianza']) * 100:.1f}% de la varianza total "
    f"({met['pca_varianza'][0] * 100:.1f}% y {met['pca_varianza'][1] * 100:.1f}% en los dos "
    "primeros componentes)."))
filas_res = [["C", "Nombre del clúster", "n", "%", "Puntaje", "Palabras", "% neg.", "% mayús."]]
for _, r in res.iterrows():
    filas_res.append([f"C{int(r.cluster)}", r.nombre, f"{r.n:,}", f"{r.pct:.1f}",
                      f"{r.puntaje_prom:.2f}", f"{r.palabras_prom:.0f}",
                      f"{r.pct_negativas:.1f}", f"{r.ratio_mayusculas_pct:.1f}"])
H.append(tabla(filas_res, anchos=[1.1 * cm, 4.9 * cm, 1.8 * cm, 1.3 * cm, 1.7 * cm, 1.8 * cm,
                                  1.6 * cm, 1.6 * cm], fs=7.9))
H.append(p("Tabla 7. Resumen comparativo de los cinco clústeres.", "caption"))
H.append(imagen(FIG / "dispersion_pca.png", 14,
                titulo="Figura 7. Dispersión de los agrupamientos en la proyección PCA, con los "
                       "centroides marcados con una X roja."))
H.append(imagen(FIG / "dispersion_puntaje_longitud.png", 14,
                titulo="Figura 8. Dispersión de las reseñas según puntaje y longitud, con los "
                       "centroides del modelo."))
H.append(imagen(FIG / "centroides_perfil.png", 14.5,
                titulo="Figura 9. Perfil de los centroides en desviaciones estándar."))
H.append(p("<b>Interpretación de los grupos</b>", "h2"))
H.append(p(
    f"<b>C2 · Elogio breve ({c_elogio.pct:.1f}%).</b> Es el grupo más numeroso. Reúne reseñas "
    f"cortas (≈{c_elogio.palabras_prom:.0f} palabras) con puntaje casi perfecto "
    f"({c_elogio.puntaje_prom:.2f}), pocos aspectos mencionados y una proporción de negatividad "
    "casi nula. Son opiniones positivas, entusiastas y poco argumentadas: valoran la experiencia "
    "general sin entrar en detalles técnicos."))
H.append(p(
    f"<b>C0 · Rechazo directo ({c_rechazo.pct:.1f}%).</b> Reseñas muy breves "
    f"(≈{c_rechazo.palabras_prom:.0f} palabras) con puntaje mínimo ({c_rechazo.puntaje_prom:.2f}) "
    f"y {c_rechazo.pct_negativas:.1f}% de valoraciones negativas. La negatividad léxica es baja, "
    "lo que sugiere un rechazo categórico y escueto, expresado con una calificación dura más que "
    "con un desarrollo argumental. Es el segmento de respuesta más visceral."))
H.append(p(
    f"<b>C4 · Crítica negativa argumentada ({c_critica.pct:.1f}%).</b> Puntaje bajo "
    f"({c_critica.puntaje_prom:.2f}) pero con mayor carga de palabras negativas "
    f"({c_critica.ratio_negativas_pct:.2f}%) y más aspectos mencionados "
    f"({c_critica.aspectos_prom:.2f}). Son reseñas breves que identifican problemas concretos "
    f"({c_critica.pct_negativas:.1f}% negativas), útiles para detectar molestias específicas."))
H.append(p(
    f"<b>C1 · Análisis detallado y crítico ({c_detalle.pct:.1f}%).</b> Es el grupo con mayor "
    f"extensión (≈{c_detalle.palabras_prom:.0f} palabras) y amplitud temática "
    f"({c_detalle.aspectos_prom:.2f} aspectos en promedio). Su puntaje medio es intermedio "
    f"({c_detalle.puntaje_prom:.2f}) porque conviven críticas y elogios: {c_detalle.pct_negativas:.1f}% "
    "puntúa 4 o menos. En este grupo, 78,8% menciona historia y ritmo, 61,0% personajes, 39,2% "
    "rendimiento y 34,1% bugs. Es el segmento con mayor valor informativo para priorizar mejoras."))
H.append(p(
    f"<b>C3 · Protesta intensiva ({c_protesta.pct:.1f}%).</b> Grupo pequeño pero distintivo: "
    f"{c_protesta.ratio_mayusculas_pct:.1f}% del texto está en mayúsculas (frente a ≈2,5% en los "
    f"demás), con puntaje {c_protesta.puntaje_prom:.2f} y {c_protesta.pct_negativas:.1f}% de "
    "reseñas negativas. Expresa malestar con alta carga emocional y funciona como señal "
    "temprana de conflicto o de dinámicas de review-bombing."))
H.append(imagen(FIG / "boxplot_puntaje_cluster.png", 14,
                titulo="Figura 10. Distribución del puntaje dentro de cada clúster."))
H.append(imagen(FIG / "aspectos_por_cluster.png", 15,
                titulo="Figura 11. Porcentaje de reseñas de cada clúster que menciona cada aspecto."))
H.append(p(
    "La Figura 11 confirma que el clúster de análisis detallado concentra las menciones de casi "
    "todos los aspectos, mientras que los grupos de rechazo directo y elogio breve mencionan "
    "pocos temas. Esto refuerza la idea de que la extensión y la amplitud temática son ejes "
    "independientes del puntaje: una reseña puede ser muy negativa sin explicar por qué, o muy "
    "positiva sin detallar qué funciona."))
H.append(p("<b>Hallazgos relevantes</b>", "h2"))
for texto in [
    f"<b>El promedio engaña.</b> El puntaje medio de {met['puntaje_promedio_modelo']:.2f} oculta "
    f"que {pct_rechazo:.1f}% de las reseñas pertenece a los dos grupos de rechazo (puntaje medio "
    f"menor a 3) y {c_elogio.pct:.1f}% al elogio breve. La distribución es bipolar.",
    f"<b>La crítica detallada es el activo más valioso.</b> El 20,8% de las reseñas escribe en "
    f"promedio {c_detalle.palabras_prom:.0f} palabras y menciona {c_detalle.aspectos_prom:.1f} "
    "aspectos; concentra las referencias a rendimiento, personajes, bugs y monetización.",
    f"<b>Historia y ritmo es el tema dominante.</b> Aparece en {met['aspecto_top_menciones']:,} "
    "reseñas y es el aspecto más mencionado en los cinco grupos. Le siguen personajes, gráficos "
    "y bugs. En precio y monetización, 82,4% de quienes lo mencionan puntúa 4 o menos.",
    "<b>La protesta intensiva es un indicador temprano.</b> Aunque representa solo 2,8% de las "
    "reseñas, su uso de mayúsculas (27,4%) y su tono la distinguen del resto y permiten "
    "monitorear picos de malestar.",
]:
    H.append(vineta(texto))
H.append(p("<b>Recomendaciones</b>", "h2"))
for texto in [
    "<b>Priorizar narrativa y ritmo.</b> Al ser el aspecto más discutido, conviene revisar el "
    "pacing de las misiones, la escritura de los protagonistas y la estructura de la campaña.",
    "<b>Ejecutar un plan de calidad técnica.</b> Atender bugs, caídas de FPS, problemas de la "
    "versión PC y sobrecalentamiento, temas que aparecen con fuerza en la crítica detallada.",
    "<b>Segmentar la comunicación.</b> Usar las reseñas detalladas como fuente de backlog y "
    "responder al rechazo directo con mensajes breves y concretos, sin intentar argumentar con "
    "quien no busca diálogo.",
    "<b>Monitorear la protesta intensiva.</b> Vigilar el clúster de mayúsculas y su evolución "
    "para anticipar picos de review-bombing y activar protocolos de contención.",
]:
    H.append(vineta(texto))
H.append(p("<b>Limitaciones</b>", "h2"))
for texto in [
    f"La silueta de {met['silueta_final']:.4f} indica grupos con traslape; no son segmentos "
    "naturales perfectamente separados, sino perfiles predominantes.",
    "La detección de aspectos se basa en palabras clave y puede no captar ironía, sinónimos o "
    "críticas implícitas; además, la cobertura léxica varía entre idiomas.",
    "El análisis es observacional: los grupos describen asociaciones y no permiten afirmar "
    "relaciones causales entre variables y satisfacción.",
    "La ventana temporal es de seis días y hay un fuerte desbalance hacia PlayStation 5, por lo "
    "que los resultados no deben generalizarse a toda la vida del juego ni a otras plataformas.",
    "K-Means tiende a producir grupos esféricos y es sensible al escalamiento y a la selección "
    "de variables; otras técnicas podrían revelar estructuras distintas.",
]:
    H.append(vineta(texto))

# ---------------------------------------------------------------------------
# 7. DASHBOARD
# ---------------------------------------------------------------------------
H.append(PageBreak())
H.append(p("7. Dashboard interactivo", "h1"))
H.append(p(
    "El dashboard se desarrolló con <b>Streamlit</b> y se ejecuta con "
    "<font face='DejaVu-Bold'>streamlit run dashboard/app.py</font>. Está organizado en dos "
    "páginas accesibles desde la barra lateral y cuenta con tres filtros que afectan a los "
    "indicadores y a las gráficas: <b>plataforma</b>, <b>rango de puntaje</b> y <b>clúster</b>. "
    "Todas las visualizaciones se alimentan de los archivos generados por el pipeline de Python, "
    "por lo que no existen datos introducidos manualmente."))
H.append(p(
    "<b>Página 1. Análisis general.</b> Presenta cuatro indicadores KPI (reseñas analizadas, "
    "puntaje promedio, porcentaje de reseñas negativas y palabras promedio), la distribución de "
    "puntajes por plataforma, la evolución diaria de las reseñas, la participación por "
    "plataforma, un ranking de aspectos mencionados, una tabla de estadísticas descriptivas y "
    "los primeros diez registros del dataset. Esta página permite contextualizar el corpus antes "
    "de entrar en los grupos."))
H.append(imagen(DASH / "dashboard_pagina1.png", 15.5, alto_max_cm=19.5,
                titulo="Figura 12. Página 1 del dashboard: análisis general y filtros."))
H.append(p(
    "<b>Página 2. Segmentación mediante K-Means.</b> Muestra los cuatro indicadores del modelo "
    "(K, silueta, inercia e iteraciones), la distribución de reseñas por clúster, la dispersión "
    "PCA con los centroides en rojo, el diagrama de caja del puntaje por grupo, el mapa de calor "
    "de aspectos por clúster, la tabla de estadísticas descriptivas, una tabla de distancias a "
    "los centroides, la descripción de cada grupo y una sección de hallazgos y recomendaciones. "
    "Los filtros permiten, por ejemplo, comparar solo las reseñas de Xbox o analizar únicamente "
    "el clúster de crítica detallada."))
H.append(imagen(DASH / "dashboard_pagina2_a.png", 15.5, alto_max_cm=18.5,
                titulo="Figura 13. Página 2: indicadores, distribución, dispersión PCA y "
                       "comparación de características."))
H.append(imagen(DASH / "dashboard_pagina2_b.png", 15.5, alto_max_cm=18.5,
                titulo="Figura 14. Página 2: descripción de los grupos, hallazgos y "
                       "recomendaciones."))
H.append(p(
    "Las capturas muestran el estado por defecto (todos los filtros activos). Al modificar un "
    "filtro, Streamlit recalcula los indicadores, la distribución y las gráficas en segundos. La "
    "sección de descripciones explica el significado de cada grupo y la de hallazgos resume los "
    "resultados principales, de modo que el dashboard puede utilizarse sin leer el informe "
    "completo."))

# ---------------------------------------------------------------------------
# 8. CONCLUSIONES
# ---------------------------------------------------------------------------
H.append(p("8. Conclusiones y recomendaciones", "h1"))
H.append(p(
    "El proyecto demuestra que K-Means permite convertir un corpus masivo de opiniones en "
    "perfiles accionables. La conclusión principal es que <b>el promedio de 5,02 es una síntesis "
    "engañosa</b>: la opinión está polarizada entre un elogio breve y mayoritario y un "
    "conjunto de rechazos que, juntos, representan cerca de la mitad de las reseñas. Reconocer "
    "esa estructura es más útil que resumirla en una sola cifra."))
H.append(p(
    "Los cinco grupos identificados tienen implicaciones distintas. El elogio breve confirma la "
    "recepción positiva de la experiencia general; el rechazo directo exige respuestas simples y "
    "no argumentativas; la crítica argumentada y el análisis detallado concentran la información "
    "técnica y temática que permite priorizar parches; y la protesta intensiva funciona como "
    "indicador temprano de conflictos. La historia y el ritmo aparecen como el eje transversal "
    "más importante, seguido por personajes, rendimiento y bugs."))
H.append(p(
    "El dashboard cumple el objetivo de comunicar los resultados de forma interactiva. Sus dos "
    "páginas permiten pasar del contexto general al detalle de los clústeres, filtrar por "
    "plataforma, puntaje y grupo, y consultar tanto las estadísticas como la interpretación de "
    "cada segmento. La tabla de distancias y la dispersión PCA hacen explícito el criterio de "
    "asignación de K-Means, lo que aporta transparencia al análisis."))
H.append(p(
    "Como trabajo futuro se propone incorporar análisis de sentimiento a nivel de frase, "
    "modelado de tópicos (por ejemplo, BERTopic) para enriquecer la detección de aspectos, "
    "comparar K-Means con DBSCAN o clustering jerárquico, ampliar la ventana temporal y "
    "equilibrar la muestra entre plataformas. También sería valioso validar los perfiles con "
    "expertos de la industria y automatizar la actualización del dashboard con nuevos datos."))
H.append(Spacer(1, 0.3 * cm))
H.append(p(
    "En síntesis, la segmentación no reemplaza la lectura de las reseñas: la organiza. Su valor "
    "está en señalar dónde mirar primero y en ofrecer una base reproducible para que las "
    "decisiones de producto y de comunicación se apoyen en evidencia y no en impresiones.", "body"))

doc.build(H, onFirstPage=pie_pagina, onLaterPages=pie_pagina)
print("PDF generado:", DOCS / "informe.pdf")
