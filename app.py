# Evaluación 2 - Teoría de Grafos
# TSP: búsqueda exhaustiva vs Vecino Más Cercano
# Distancias euclidianas sobre el plano (latitud, longitud)

import math
import itertools
import time
import os

import matplotlib.pyplot as plt
from staticmap import StaticMap, CircleMarker, Line  # para mapas con fondo real

# =====================================
# 1. DATOS: CIUDADES Y COORDENADAS
# =====================================

ciudades = {
    "Temuco":       (-38.7397,   -72.5984),
    "Villarrica":   (-39.2670,   -72.2170),
    "Valdivia":     (-39.819588, -73.245209),
    "Osorno":       (-40.574505, -73.131920),
    "Frutillar":    (-41.12278,  -73.05806),
    "Puerto Varas": (-41.3170,   -72.9830),
    "Puerto Montt": (-41.4670,   -72.9330),
}

nombres = [
    "Temuco",
    "Villarrica",
    "Valdivia",
    "Osorno",
    "Frutillar",
    "Puerto Varas",
    "Puerto Montt"
]

coords = [ciudades[n] for n in nombres]
n = len(nombres)

# =====================================
# 2. MATRIZ DE DISTANCIAS (EUCLIDIANA EN GRADOS)
# =====================================

def distancia_euclidea(i, j):
    """Distancia euclidiana sobre el plano (lat, lon), unidades: grados."""
    lat1, lon1 = coords[i]
    lat2, lon2 = coords[j]
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

D = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        D[i][j] = distancia_euclidea(i, j)

def imprimir_matriz_distancias():
    print("\n===== MATRIZ DE DISTANCIAS (EUCLIDIANA, UNIDADES: GRADOS) =====\n")
    print(f"{'Ciudad':>12}", end="")
    for name in nombres:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, name in enumerate(nombres):
        print(f"{name[:12]:>12}", end="")
        for j in range(n):
            print(f"{D[i][j]:12.3f}", end="")
        print()
    print()

# =====================================
# 3. TSP: FUNCIONES COMUNES
# =====================================

def longitud_ciclo(ruta, D):
    total = 0.0
    for k in range(len(ruta) - 1):
        total += D[ruta[k]][ruta[k+1]]
    total += D[ruta[-1]][ruta[0]]
    return total

def tsp_exhaustivo(D, inicio=0):
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
    return mejor_ruta, mejor_longitud, t1 - t0

def tsp_vecino_mas_cercano(D, inicio=0):
    n = len(D)
    no_visitadas = set(range(n))
    ruta = [inicio]
    no_visitadas.remove(inicio)

    t0 = time.perf_counter()
    actual = inicio
    while no_visitadas:
        siguiente = min(no_visitadas, key=lambda j: D[actual][j])
        ruta.append(siguiente)
        no_visitadas.remove(siguiente)
        actual = siguiente
    t1 = time.perf_counter()

    return ruta, longitud_ciclo(ruta, D), t1 - t0

# =====================================
# 4. GRÁFICOS (PLANO LAT-LON)
# =====================================

def guardar_ruta_como_imagen(ruta, nombres, coords, archivo, titulo):
    xs = [coords[i][1] for i in ruta]
    ys = [coords[i][0] for i in ruta]
    xs.append(xs[0])
    ys.append(ys[0])

    plt.figure()
    plt.plot(xs, ys, '-o')
    for i in ruta:
        plt.text(coords[i][1], coords[i][0], nombres[i], fontsize=9)

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(archivo, dpi=300)
    plt.close()
    print(f"Imagen guardada en: {archivo}")

# =====================================
# 5. MAPA REAL (OSM) CON STATICMAP
# =====================================

def guardar_ruta_como_mapa(ruta, nombres, coords, archivo, size=(800, 800), zoom=7):
    m = StaticMap(size[0], size[1])

    for i in ruta:
        lat, lon = coords[i]
        m.add_marker(CircleMarker((lon, lat), 'red', 10))

    puntos = [(coords[i][1], coords[i][0]) for i in ruta]
    puntos.append((coords[ruta[0]][1], coords[ruta[0]][0]))
    m.add_line(Line(puntos, 'blue', 4))

    img = m.render(zoom=zoom)
    img.save(archivo)
    print(f"Mapa guardado en: {archivo}")

# =====================================
# 6. MAIN
# =====================================

if __name__ == "__main__":
    imprimir_matriz_distancias()

    inicio = 0  # Temuco

    ruta_opt, L_opt, t_exh = tsp_exhaustivo(D, inicio)
    ruta_nn, L_nn, t_nn = tsp_vecino_mas_cercano(D, inicio)
    gap = (L_nn - L_opt) / L_opt * 100

    print("=== RESULTADOS (UNIDADES: GRADOS) ===")
    print("Ruta óptima:", " → ".join(nombres[i] for i in ruta_opt), "→", nombres[ruta_opt[0]])
    print("L* = {:.4f}".format(L_opt))
    print("Tiempo exhaustivo =", t_exh, "s\n")

    print("Ruta NN:", " → ".join(nombres[i] for i in ruta_nn), "→", nombres[ruta_nn[0]])
    print("L_NN = {:.4f}".format(L_nn))
    print("Tiempo NN =", t_nn, "s\n")

    print("Gap = {:.2f}%".format(gap))

    guardar_ruta_como_imagen(ruta_opt, nombres, coords, "ruta_optima.png", "Ruta óptima (exhaustivo)")
    guardar_ruta_como_imagen(ruta_nn, nombres, coords, "ruta_nn.png", "Ruta heurística NN")

    guardar_ruta_como_mapa(ruta_opt, nombres, coords, "mapa_optimo.png")
    guardar_ruta_como_mapa(ruta_nn, nombres, coords, "mapa_nn.png")
