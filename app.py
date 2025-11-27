# Evaluación 2 - Teoría de Grafos
# TSP: búsqueda exhaustiva vs Vecino Más Cercano
# Orden de ciudades:
# Temuco, Villarrica, Valdivia, Osorno, Frutillar, Puerto Varas, Puerto Montt

import math
import itertools
import time
import os

import matplotlib.pyplot as plt
import folium  # para mapas interactivos

# =====================================
# 1. DATOS: CIUDADES Y COORDENADAS
# =====================================

# Diccionario: nombre -> (latitud, longitud)
ciudades = {
    "Temuco":       (-38.7397,   -72.5984),
    "Villarrica":   (-39.2670,   -72.2170),
    "Valdivia":     (-39.819588, -73.245209),
    "Osorno":       (-40.574505, -73.131920),
    "Frutillar":    (-41.12278,  -73.05806),
    "Puerto Varas": (-41.3170,   -72.9830),
    "Puerto Montt": (-41.4670,   -72.9330),
}

# Orden FIJO de las ciudades (el que usas en las tablas y el informe)
nombres = [
    "Temuco",
    "Villarrica",
    "Valdivia",
    "Osorno",
    "Frutillar",
    "Puerto Varas",
    "Puerto Montt"
]

# Vector de coordenadas respetando ese orden
coords = [ciudades[n] for n in nombres]
n = len(nombres)

# =====================================
# 2. MATRIZ DE DISTANCIAS D
# =====================================

def distancia(i, j):
    """Distancia euclidiana entre ciudad i y j (índices sobre 'coords')."""
    lat1, lon1 = coords[i]
    lat2, lon2 = coords[j]
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

# Construir matriz D
D = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        D[i][j] = distancia(i, j)

def imprimir_matriz_distancias():
    """Imprime la matriz D en formato tabular (para revisar en consola)."""
    print("Matriz de distancias D (unidades: grados lat/long):")
    print(" " * 12, end="")
    for name in nombres:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, name in enumerate(nombres):
        print(f"{name[:10]:>12}", end="")
        for j in range(n):
            print(f"{D[i][j]:12.3f}", end="")
        print()
    print()

# =====================================
# 3. FUNCIONES COMUNES
# =====================================

def longitud_ciclo(ruta, D):
    """
    Calcula la longitud total de un ciclo Hamiltoniano cerrado.
    'ruta' es una lista de índices de ciudades en orden de visita (sin repetir inicio).
    """
    total = 0.0
    for i in range(len(ruta) - 1):
        total += D[ruta[i]][ruta[i+1]]
    # cerrar ciclo volviendo al inicio
    total += D[ruta[-1]][ruta[0]]
    return total

# =====================================
# 4. BÚSQUEDA EXHAUSTIVA
# =====================================

def tsp_exhaustivo(D, inicio=0):
    """
    Búsqueda exhaustiva (fuerza bruta).
    Devuelve (mejor_ruta_indices, longitud_optima, tiempo_segundos).
    """
    n = len(D)
    otros = [i for i in range(n) if i != inicio]

    mejor_ruta = None
    mejor_longitud = float("inf")

    t0 = time.perf_counter()

    for perm in itertools.permutations(otros):
        ruta = [inicio] + list(perm)
        L = longitud_ciclo(ruta, D)
        if L < mejor_longitud:
            mejor_longitud = L
            mejor_ruta = ruta

    t1 = time.perf_counter()
    tiempo = t1 - t0

    return mejor_ruta, mejor_longitud, tiempo

# =====================================
# 5. VECINO MÁS CERCANO (NN)
# =====================================

def tsp_vecino_mas_cercano(D, inicio=0):
    """
    Heurística de Vecino Más Cercano.
    Devuelve (ruta_indices, longitud_total, tiempo_segundos).
    """
    n = len(D)
    no_visitadas = set(range(n))
    ruta = [inicio]
    no_visitadas.remove(inicio)

    t0 = time.perf_counter()

    actual = inicio
    while no_visitadas:
        # ciudad no visitada más cercana
        siguiente = min(no_visitadas, key=lambda j: D[actual][j])
        ruta.append(siguiente)
        no_visitadas.remove(siguiente)
        actual = siguiente

    L = longitud_ciclo(ruta, D)

    t1 = time.perf_counter()
    tiempo = t1 - t0

    return ruta, L, tiempo

# =====================================
# 6. VISUALIZACIÓN CON MATPLOTLIB
# =====================================

def plot_ruta(ruta, nombres, coords, titulo="Ruta TSP"):
    """
    Dibuja una ruta cerrada sobre las coordenadas de las ciudades (sin mapa de fondo).
    """
    xs = [coords[i][1] for i in ruta]  # longitud eje X
    ys = [coords[i][0] for i in ruta]  # latitud eje Y

    # cerrar ciclo
    xs.append(xs[0])
    ys.append(ys[0])

    plt.figure()
    plt.plot(xs, ys, marker="o", linestyle="-")
    for i in ruta:
        lon = coords[i][1]
        lat = coords[i][0]
        plt.text(lon, lat, " " + nombres[i], fontsize=9, va="center")

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =====================================
# 7. MAPA REAL CON FOLIUM
# =====================================

def crear_mapa_ruta(ruta, nombres, coords, nombre_archivo="mapa_ruta.html", titulo="Ruta TSP"):
    """
    Crea un mapa interactivo (OpenStreetMap) con la ruta dibujada.
    - ruta: lista de índices de ciudades en orden de visita.
    - nombre_archivo: archivo HTML donde se guardará el mapa.
    """
    # centro del mapa: promedio de coordenadas
    lat_c = sum(lat for lat, lon in coords) / len(coords)
    lon_c = sum(lon for lat, lon in coords) / len(coords)

    mapa = folium.Map(location=[lat_c, lon_c], zoom_start=8, tiles="OpenStreetMap")

    # marcar las ciudades
    for i, (lat, lon) in enumerate(coords):
        folium.Marker(
            location=[lat, lon],
            popup=nombres[i],
            tooltip=nombres[i]
        ).add_to(mapa)

    # construir la polilínea de la ruta (incluye vuelta al inicio)
    puntos = []
    for idx in ruta:
        lat, lon = coords[idx]
        puntos.append((lat, lon))
    # cerrar ciclo
    puntos.append((coords[ruta[0]][0], coords[ruta[0]][1]))

    folium.PolyLine(
        locations=puntos,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip=titulo
    ).add_to(mapa)

    mapa.save(nombre_archivo)
    print(f"Mapa guardado en: {nombre_archivo}")

# =====================================
# 8. VISUALIZACIÓN PASO A PASO NN (PNG)
# =====================================

def plot_ruta_parcial(ruta, nombres, coords, paso, carpeta="fig_nn_pasos"):
    """
    Dibuja la ruta parcial en el paso dado y la guarda como imagen PNG.
    """
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    xs = [coords[i][1] for i in ruta]
    ys = [coords[i][0] for i in ruta]

    # si ya están todas las ciudades, cerramos ciclo
    if len(ruta) == len(coords):
        xs.append(xs[0])
        ys.append(ys[0])

    plt.figure()
    plt.plot(xs, ys, marker="o", linestyle="-")
    for i in ruta:
        lon = coords[i][1]
        lat = coords[i][0]
        plt.text(lon, lat, " " + nombres[i], fontsize=9, va="center")

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(f"Vecino Más Cercano - Paso {paso}")
    plt.grid(True)
    plt.tight_layout()

    archivo = os.path.join(carpeta, f"nn_paso_{paso:02d}.png")
    plt.savefig(archivo)
    plt.close()

def tsp_vecino_mas_cercano_con_visualizacion(D, nombres, coords, inicio=0):
    """
    Igual que tsp_vecino_mas_cercano, pero guarda una imagen por cada paso.
    Devuelve (ruta_indices, longitud_total).
    """
    n = len(D)
    no_visitadas = set(range(n))
    ruta = [inicio]
    no_visitadas.remove(inicio)

    paso = 1
    plot_ruta_parcial(ruta, nombres, coords, paso)

    actual = inicio
    while no_visitadas:
        siguiente = min(no_visitadas, key=lambda j: D[actual][j])
        ruta.append(siguiente)
        no_visitadas.remove(siguiente)
        actual = siguiente
        paso += 1
        plot_ruta_parcial(ruta, nombres, coords, paso)

    L = longitud_ciclo(ruta, D)
    return ruta, L

# =====================================
# 9. PROGRAMA PRINCIPAL
# =====================================

if __name__ == "__main__":
    print("Ciudades en el orden utilizado:")
    for i, nombre in enumerate(nombres):
        print(f"  {i}: {nombre}")
    print()

    imprimir_matriz_distancias()

    # Ciudad inicial: Temuco (índice 0 en 'nombres')
    inicio = nombres.index("Temuco")

    # --- Búsqueda exhaustiva ---
    ruta_opt, L_opt, t_exh = tsp_exhaustivo(D, inicio=inicio)

    # --- Vecino Más Cercano ---
    ruta_nn, L_nn, t_nn = tsp_vecino_mas_cercano(D, inicio=inicio)

    # --- Comparación ---
    gap = (L_nn - L_opt) / L_opt * 100

    print("=== RESULTADOS ===")
    print("Ruta óptima (búsqueda exhaustiva):")
    print("  Índices:", ruta_opt)
    print("  Nombres:", " → ".join(nombres[i] for i in ruta_opt), "→", nombres[ruta_opt[0]])
    print(f"  Longitud L*: {L_opt:.4f}")
    print(f"  Tiempo búsqueda exhaustiva: {t_exh:.6f} s\n")

    print("Ruta heurística (Vecino Más Cercano):")
    print("  Índices:", ruta_nn)
    print("  Nombres:", " → ".join(nombres[i] for i in ruta_nn), "→", nombres[ruta_nn[0]])
    print(f"  Longitud L_NN: {L_nn:.4f}")
    print(f"  Tiempo heurística NN: {t_nn:.6f} s\n")

    print("Comparación:")
    print(f"  Gap de optimalidad g = ((L_NN - L*) / L*) * 100 = {gap:.2f} %")

    # --- Gráficos de las rutas (sin mapa de fondo) ---
    plot_ruta(ruta_opt, nombres, coords, titulo="Ciclo óptimo (búsqueda exhaustiva)")
    plot_ruta(ruta_nn, nombres, coords, titulo="Ciclo heurístico (Vecino Más Cercano)")

    # --- Mapas reales con Folium ---
    crear_mapa_ruta(
        ruta_opt,
        nombres,
        coords,
        nombre_archivo="mapa_ruta_optima.html",
        titulo="Ruta óptima (búsqueda exhaustiva)"
    )

    crear_mapa_ruta(
        ruta_nn,
        nombres,
        coords,
        nombre_archivo="mapa_ruta_vecino_mas_cercano.html",
        titulo="Ruta heurística (Vecino Más Cercano)"
    )

    # --- Imágenes paso a paso de NN (opcional) ---
    # ruta_nn_pas, L_nn_pas = tsp_vecino_mas_cercano_con_visualizacion(D, nombres, coords, inicio=inicio)
    # print("Imágenes del proceso NN guardadas en carpeta 'fig_nn_pasos/'")
