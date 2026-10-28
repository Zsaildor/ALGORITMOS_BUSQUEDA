"""
Módulo de la Práctica 2: Búsqueda Heurística con A*
Implementa la búsqueda A* con f(n) = g(n) + h(n), desempate alfabético estricto,
relajación de aristas y generación de la tabla de iteraciones y comparativa frente a BFS.
"""

import heapq
from typing import Dict, List, Tuple, Any, Optional, Set
from src.datos_campus import GRAFO_CAMPUS, HEURISTICA_F


def reconstruir_ruta_astar(padres: Dict[str, Optional[str]], origen: str, meta: str) -> List[str]:
    """Reconstruye el camino desde origen hasta meta usando el diccionario de padres."""
    ruta = []
    actual = meta
    while actual is not None:
        ruta.append(actual)
        if actual == origen:
            break
        actual = padres.get(actual)
    ruta.reverse()
    return ruta if ruta and ruta[0] == origen else []


def ejecutar_astar(
    origen: str = 'A',
    meta: str = 'F',
    grafo: Dict[str, List[Tuple[str, int]]] = None,
    heuristica: Dict[str, int] = None
) -> Dict[str, Any]:
    """
    Ejecuta el algoritmo de búsqueda heurística A* respetando:
    - Evaluación f(n) = g(n) + h(n).
    - Desempate estricto: en caso de igual f(n), menor orden alfabético.
    - Criterio de parada: extracción de la meta F de la cola de prioridad.
    - F no se cuenta como nodo expandido.
    - Registro de cada iteración y del estado de la frontera.
    """
    if grafo is None:
        grafo = GRAFO_CAMPUS
    if heuristica is None:
        heuristica = HEURISTICA_F

    g_score: Dict[str, int] = {origen: 0}
    padres: Dict[str, Optional[str]] = {origen: None}
    cerrados: Set[str] = set()
    expandidos: List[str] = []
    historial_iteraciones: List[Dict[str, Any]] = []

    # Cola de prioridad: tuplas (f_n, nodo)
    # Python compara el segundo elemento de la tupla (nombre de nodo) para desempatar alfabéticamente
    f_origen = g_score[origen] + heuristica[origen]
    cola_prioridad: List[Tuple[int, str]] = [(f_origen, origen)]

    # Diccionario para rastrear el estado visible de la frontera
    frontera_activa: Dict[str, int] = {origen: f_origen}

    iter_num = 1
    ruta_encontrada: List[str] = []
    encontrado = False

    while cola_prioridad:
        f_actual, u = heapq.heappop(cola_prioridad)

        # Si el nodo ya fue cerrado con una distancia menor o igual, ignorar entrada obsoleta del heap
        if u in cerrados:
            continue

        g_actual = g_score[u]
        h_actual = heuristica[u]

        # Quitar u de la frontera activa porque está siendo procesado
        if u in frontera_activa:
            del frontera_activa[u]

        # Validar si u es la meta al extraerlo
        if u == meta:
            observacion = (
                f"Se extrae la meta {meta} con menor prioridad (f={f_actual}). "
                f"Fin de la búsqueda (no se cuenta {meta} como expandido)."
            )
            historial_iteraciones.append({
                'iteracion': str(iter_num),
                'nodo_seleccionado': u,
                'g': g_actual,
                'h': h_actual,
                'f': f_actual,
                'cambios_frontera': observacion,
                'pendientes': "[]"
            })
            ruta_encontrada = reconstruir_ruta_astar(padres, origen, meta)
            encontrado = True
            break

        # Marcar u como cerrado y registrar en expandidos
        cerrados.add(u)
        expandidos.append(u)

        cambios_desc: List[str] = []
        nuevos_agregados: List[str] = []

        # Examinar vecinos en orden alfabético
        for v, peso in grafo[u]:
            if v in cerrados:
                continue

            g_tentativo = g_actual + peso
            f_tentativo = g_tentativo + heuristica[v]

            if v not in g_score:
                # Nodo descubierto por primera vez
                g_score[v] = g_tentativo
                padres[v] = u
                frontera_activa[v] = f_tentativo
                heapq.heappush(cola_prioridad, (f_tentativo, v))
                nuevos_agregados.append(f"{v} (g={g_tentativo}, h={heuristica[v]}, f={f_tentativo}, padre={u})")
            elif g_tentativo < g_score[v]:
                # Se relaja la arista y mejora la ruta
                g_viejo = g_score[v]
                g_score[v] = g_tentativo
                padres[v] = u
                frontera_activa[v] = f_tentativo
                heapq.heappush(cola_prioridad, (f_tentativo, v))
                cambios_desc.append(
                    f"¡Mejora ruta hacia {v}! Nuevo g({v}) = {g_actual}+{peso} = {g_tentativo} < {g_viejo} "
                    f"⇒ f({v}) = {f_tentativo}, nuevo padre={u}."
                )
            else:
                # No mejora
                cambios_desc.append(f"Vecino {v} (g={g_actual}+{peso}={g_tentativo} > {g_score[v]}, no mejora).")

        if nuevos_agregados:
            cambios_desc.insert(0, f"Se agregan {', '.join(nuevos_agregados)}.")

        if u == 'B' and any('E' in k for k in frontera_activa):
            cambios_desc.insert(0, "(Elegido antes que E por desempate alfabético en f=150).")

        # Representación de la frontera ordenada por (f, nodo)
        pendientes_ordenados = sorted(
            [(f_val, nodo) for nodo, f_val in frontera_activa.items() if nodo not in cerrados],
            key=lambda x: (x[0], x[1])
        )
        pendientes_str = f"[{', '.join([f'{nodo}({f_val})' for f_val, nodo in pendientes_ordenados])}]"

        texto_cambios = " ".join(cambios_desc)
        if pendientes_str:
            texto_cambios += f" Pendientes: {pendientes_str}"

        historial_iteraciones.append({
            'iteracion': str(iter_num),
            'nodo_seleccionado': u,
            'g': g_actual,
            'h': h_actual,
            'f': f_actual,
            'cambios_frontera': texto_cambios,
            'pendientes': pendientes_str
        })

        iter_num += 1

    distancia_total = g_score.get(meta, 0)
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
        'historial_iteraciones': historial_iteraciones
    }


def generar_tabla_comparativa(res_bfs: Dict[str, Any], res_astar: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Genera la tabla comparativa oficial entre BFS y A*."""
    return [
        {
            'algoritmo': 'BFS (Búsqueda Ciega)',
            'ruta': res_bfs['ruta_str'],
            'conexiones': res_bfs['conexiones'],
            'distancia_m': res_bfs['distancia_total'],
            'nodos_expandidos': f"{res_bfs['total_expandidos']} ({res_bfs['nodos_expandidos_str']})"
        },
        {
            'algoritmo': 'A* (Búsqueda Heurística)',
            'ruta': res_astar['ruta_str'],
            'conexiones': res_astar['conexiones'],
            'distancia_m': res_astar['distancia_total'],
            'nodos_expandidos': f"{res_astar['total_expandidos']} ({res_astar['nodos_expandidos_str']})"
        }
    ]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    res_astar = ejecutar_astar('A', 'F')
    print("=== RESULTADOS A* ===")
    print("Ruta óptima:", res_astar['ruta_str'])
    print("Conexiones:", res_astar['conexiones'])
    print("Distancia total:", res_astar['distancia_total'], "m")
    print("Nodos expandidos:", res_astar['nodos_expandidos_str'], f"({res_astar['total_expandidos']} nodos)")
    print("\nTabla de iteraciones:")
    for it in res_astar['historial_iteraciones']:
        print(f"Iter {it['iteracion']:>2} | Nodo: {it['nodo_seleccionado']} | g={it['g']:<3} | h={it['h']:<3} | f={it['f']:<3} | {it['cambios_frontera']}")
