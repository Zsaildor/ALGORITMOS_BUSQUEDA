"""
Módulo de la Práctica 1: Búsqueda Ciega con BFS (Breadth-First Search)
Implementa la búsqueda en anchura para minimizar conexiones entre un origen y una meta,
generando la tabla de seguimiento paso a paso y las métricas finales.
"""

from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from src.datos_campus import GRAFO_CAMPUS


def calcular_distancia_ruta(ruta: List[str], grafo: Dict[str, List[Tuple[str, int]]]) -> int:
    """Calcula la distancia física total en metros sumando los pesos de cada tramo en la ruta."""
    distancia = 0
    for i in range(len(ruta) - 1):
        u, v = ruta[i], ruta[i + 1]
        encontrado = False
        for vecino, peso in grafo[u]:
            if vecino == v:
                distancia += peso
                encontrado = True
                break
        if not encontrado:
            raise ValueError(f"No existe arista directa entre {u} y {v} en el grafo.")
    return distancia


def reconstruir_ruta(padres: Dict[str, Optional[str]], origen: str, meta: str) -> List[str]:
    """Reconstruye el camino desde el origen hasta la meta siguiendo el diccionario de padres."""
    ruta = []
    actual = meta
    while actual is not None:
        ruta.append(actual)
        if actual == origen:
            break
        actual = padres.get(actual)
    ruta.reverse()
    return ruta if ruta and ruta[0] == origen else []


def ejecutar_bfs(origen: str = 'A', meta: str = 'F', grafo: Dict[str, List[Tuple[str, int]]] = None) -> Dict[str, Any]:
    """
    Ejecuta el algoritmo de búsqueda en anchura (BFS) respetando:
    - Exploración de vecinos en estricto orden alfabético.
    - Marcado de descubierto en el momento de encolar.
    - Criterio de parada al extraer la meta (la meta NO se contabiliza como expandida).
    - Registro detallado de la traza paso a paso para la interfaz gráfica.
    """
    if grafo is None:
        grafo = GRAFO_CAMPUS

    cola: deque = deque([origen])
    descubiertos: set = {origen}
    padres: Dict[str, Optional[str]] = {origen: None}
    expandidos: List[str] = []
    historial_pasos: List[Dict[str, str]] = []

    # Registro de estado Inicial
    historial_pasos.append({
        'paso': 'Inicial',
        'nodo_procesado': '—',
        'cola': f"[{', '.join(cola)}]",
        'descubiertos': f"{{{', '.join(sorted(list(descubiertos)))}}}",
        'observaciones': f"Padre[{origen}] = null"
    })

    paso_num = 1
    ruta_encontrada: List[str] = []
    encontrado = False

    while cola:
        u = cola.popleft()

        # Validación de prueba de meta al extraer de la cola
        if u == meta:
            historial_pasos.append({
                'paso': str(paso_num),
                'nodo_procesado': u,
                'cola': f"[{', '.join(cola)}]",
                'descubiertos': f"{{{', '.join(sorted(list(descubiertos)))}}}",
                'observaciones': f"Se extrae la meta {meta} y termina (no se expande {meta})"
            })
            ruta_encontrada = reconstruir_ruta(padres, origen, meta)
            encontrado = True
            break

        # Contabilizar como nodo expandido (procesamiento de sus vecinos)
        expandidos.append(u)
        vecinos_descubiertos_este_paso = []

        # Los vecinos ya están en orden alfabético en el grafo
        for v, _ in grafo[u]:
            if v not in descubiertos:
                descubiertos.add(v)
                padres[v] = u
                cola.append(v)
                vecinos_descubiertos_este_paso.append(v)

        # Redactar observación del paso
        if vecinos_descubiertos_este_paso:
            if meta in vecinos_descubiertos_este_paso:
                desc_str = ", ".join(vecinos_descubiertos_este_paso)
                obs = f"Se descubre {desc_str} desde {u} (Padre[{meta}] = {u})"
            else:
                desc_str = " y ".join(vecinos_descubiertos_este_paso)
                obs = f"Se descubren {desc_str} desde {u}"
        else:
            obs = f"Vecinos de {u} ya estaban descubiertos"

        historial_pasos.append({
            'paso': str(paso_num),
            'nodo_procesado': u,
            'cola': f"[{', '.join(cola)}]",
            'descubiertos': f"{{{', '.join(sorted(list(descubiertos)))}}}",
            'observaciones': obs
        })

        paso_num += 1

    distancia_total = calcular_distancia_ruta(ruta_encontrada, grafo) if ruta_encontrada else 0
    conexiones = max(0, len(ruta_encontrada) - 1)

    return {
        'origen': origen,
        'meta': meta,
        'encontrado': encontrado,
        'ruta': ruta_encontrada,
        'ruta_str': " → ".join(ruta_encontrada),
        'conexiones': conexiones,
        'distancia_total': distancia_total,
        'nodos_expandidos': expandidos,
        'nodos_expandidos_str': ", ".join(expandidos),
        'total_expandidos': len(expandidos),
        'historial_pasos': historial_pasos
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    resultado = ejecutar_bfs('A', 'F')
    print("=== RESULTADOS BFS ===")
    print("Ruta:", resultado['ruta_str'])
    print("Conexiones:", resultado['conexiones'])
    print("Distancia total:", resultado['distancia_total'], "m")
    print("Nodos expandidos:", resultado['nodos_expandidos_str'], f"({resultado['total_expandidos']} nodos)")
    print("\nTabla de seguimiento:")
    for paso in resultado['historial_pasos']:
        print(f"Paso {paso['paso']:>7} | Nodo: {paso['nodo_procesado']} | Cola: {paso['cola']:<12} | Descubiertos: {paso['descubiertos']:<22} | Obs: {paso['observaciones']}")
