"""Captura las dos páginas del dashboard para el informe PDF.

Requiere el dashboard en ejecución (streamlit run dashboard/app.py).
Uso: python src/capturas_dashboard.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "reports" / "dashboard"
SALIDA.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    navegador = p.chromium.launch()
    pagina = navegador.new_page(viewport={"width": 1600, "height": 1800}, device_scale_factor=1)
    alturas = {1: 1800, 2: 3650}
    for num, espera in [(1, "Reseñas analizadas"), (2, "Distribución de reseñas por clúster")]:
        pagina.set_viewport_size({"width": 1600, "height": alturas[num]})
        pagina.goto(f"http://localhost:8501/?pagina={num}", wait_until="networkidle")
        pagina.wait_for_selector(f"text={espera}", timeout=60000)
        pagina.wait_for_timeout(8000)
        destino = SALIDA / f"dashboard_pagina{num}.png"
        pagina.screenshot(path=str(destino))
        print("Guardada:", destino)
    navegador.close()
