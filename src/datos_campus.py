"""
Módulo de Datos del Campus Universitario
Contiene el catálogo de nodos, conexiones con pesos positivos,
lista de adyacencia ordenada alfabéticamente, función de heurística h(n)
y cálculo de caminos mínimos con Dijkstra.
"""

import heapq
from typing import Dict, List, Tuple, Optional

# Catálogo oficial de nodos del campus
NODOS_CAMPUS: Dict[str, str] = {
    'A': 'Almacén central',
    'B': 'Biblioteca',
    'C': 'Edificio de aulas',
    'D': 'Cafetería',
    'E': 'Servicios escolares',
    'F': 'Laboratorio de cómputo'
}

# Conexiones bidireccionales con sus pesos en metros
ARISTAS_DEFINIDAS: List[Tuple[str, str, int]] = [
    ('A', 'B', 60),
    ('A', 'C', 40),
    ('B', 'D', 50),
    ('B', 'E', 90),
    ('C', 'D', 30),
    ('C', 'E', 80),
    ('D', 'F', 100),
    ('E', 'F', 40),
]

# Valores de heurística h(n) hacia el Laboratorio de cómputo (F)
HEURISTICA_F: Dict[str, int] = {
    'A': 100,
    'B': 90,
    'C': 80,
    'D': 70,
    'E': 30,
    'F': 0
}

# Coordenadas 2D para visualización limpia en NetworkX / Matplotlib
POSICIONES_NODOS: Dict[str, Tuple[float, float]] = {
    'A': (0.5, 2.0),
    'B': (2.5, 3.5),
    'C': (2.5, 0.5),
    'D': (4.5, 3.0),
    'E': (4.5, 1.0),
    'F': (6.5, 2.0)
}


def validar_pesos(aristas: List[Tuple[str, str, int]]) -> bool:
    """Verifica que todos los pesos de las aristas sean estrictamente mayores a cero."""
    for u, v, peso in aristas:
        if peso <= 0:
            raise ValueError(f"Peso inválido detectado entre {u} y {v}: {peso}. Debe ser > 0.")
    return True


def construir_grafo() -> Dict[str, List[Tuple[str, int]]]:
    """
    Construye la lista de adyacencia no dirigida con los vecinos
    ordenados estrictamente en orden alfabético.
    """
    validar_pesos(ARISTAS_DEFINIDAS)
    grafo: Dict[str, List[Tuple[str, int]]] = {nodo: [] for nodo in NODOS_CAMPUS}

    for u, v, peso in ARISTAS_DEFINIDAS:
        grafo[u].append((v, peso))
        grafo[v].append((u, peso))

    # Ordenar alfabéticamente los vecinos de cada nodo
    for nodo in grafo:
        grafo[nodo].sort(key=lambda item: item[0])

    return grafo


# Grafo global inicializado y verificado
GRAFO_CAMPUS: Dict[str, List[Tuple[str, int]]] = construir_grafo()


def dijkstra_un_origen(origen: str, grafo: Dict[str, List[Tuple[str, int]]] = None) -> Tuple[Dict[str, int], Dict[str, Optional[str]]]:
    """
    Calcula las distancias mínimas y el árbol de predecesores desde un nodo origen
    hacia todos los demás nodos usando el algoritmo de Dijkstra.
    """
    if grafo is None:
        grafo = GRAFO_CAMPUS

    distancias: Dict[str, int] = {nodo: float('inf') for nodo in grafo}
    padres: Dict[str, Optional[str]] = {nodo: None for nodo in grafo}
    distancias[origen] = 0

    cola_prioridad = [(0, origen)]

    while cola_prioridad:
        dist_u, u = heapq.heappop(cola_prioridad)

        if dist_u > distancias[u]:
            continue

        for v, peso in grafo[u]:
            dist_tentativa = dist_u + peso
            if dist_tentativa < distancias[v]:
                distancias[v] = dist_tentativa
                padres[v] = u
                heapq.heappush(cola_prioridad, (dist_tentativa, v))

    return distancias, padres


def reconstruir_camino_dijkstra(padres: Dict[str, Optional[str]], origen: str, destino: str) -> List[str]:
    """Reconstruye la lista de nodos del camino mínimo entre origen y destino."""
    if origen == destino:
        return [origen]
    camino = []
    actual = destino
    while actual is not None:
        camino.append(actual)
        if actual == origen:
            break
        actual = padres.get(actual)
    camino.reverse()
    return camino if camino and camino[0] == origen else []


def calcular_matriz_distancias_minimas() -> Tuple[Dict[str, Dict[str, int]], Dict[Tuple[str, str], List[str]]]:
    """
    Calcula la matriz completa de distancias mínimas (6x6) y los caminos físicos
    reales entre todos los pares de nodos del campus.
    """
    nodos = sorted(list(NODOS_CAMPUS.keys()))
    matriz_dist: Dict[str, Dict[str, int]] = {u: {} for u in nodos}
    rutas_fisicas: Dict[Tuple[str, str], List[str]] = {}

    for u in nodos:
        distancias, padres = dijkstra_un_origen(u)
        for v in nodos:
            matriz_dist[u][v] = distancias[v]
            rutas_fisicas[(u, v)] = reconstruir_camino_dijkstra(padres, u, v)

    return matriz_dist, rutas_fisicas


# Precalcular matriz y rutas físicas globales
MATRIZ_DISTANCIAS_MINIMAS, RUTAS_FISICAS_PARES = calcular_matriz_distancias_minimas()
