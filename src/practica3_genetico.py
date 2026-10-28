"""
Módulo de la Práctica 3: Optimización con Algoritmo Genético
Resuelve el problema de secuenciación de entregas (TSP sobre campus) visitando
los destinos [B, C, D, E, F] con origen y regreso en A, utilizando la matriz de Dijkstra.
"""

import random
import itertools
from typing import List, Tuple, Dict, Any
from src.datos_campus import (
    MATRIZ_DISTANCIAS_MINIMAS,
    RUTAS_FISICAS_PARES,
    NODOS_CAMPUS
)

DESTINOS = ['B', 'C', 'D', 'E', 'F']
SEMILLAS_OFICIALES = [42, 101, 2026, 777, 999]


def calcular_distancia_cromosoma(cromosoma: List[str], matriz_dist: Dict[str, Dict[str, int]] = None) -> int:
    """
    Calcula la distancia total del recorrido completo:
    Salida de A -> cromosoma[0] -> ... -> cromosoma[-1] -> Regreso a A.
    """
    if matriz_dist is None:
        matriz_dist = MATRIZ_DISTANCIAS_MINIMAS

    distancia = matriz_dist['A'][cromosoma[0]]
    for i in range(len(cromosoma) - 1):
        distancia += matriz_dist[cromosoma[i]][cromosoma[i + 1]]
    distancia += matriz_dist[cromosoma[-1]]['A']
    return distancia


def aptitud(distancia: int) -> float:
    """Calcula la función de aptitud a maximizar: 1 / distancia_total."""
    return 1.0 / distancia if distancia > 0 else 0.0


def generar_poblacion_inicial(tamano: int = 10, rng: random.Random = None) -> List[List[str]]:
    """
    Genera una población inicial de individuos distintos (permutaciones únicas sin repetición).
    """
    if rng is None:
        rng = random

    poblacion = []
    vistos = set()

    while len(poblacion) < tamano:
        individuo = DESTINOS.copy()
        rng.shuffle(individuo)
        clave = tuple(individuo)
        if clave not in vistos:
            vistos.add(clave)
            poblacion.append(individuo)

    return poblacion


def seleccion_torneo(poblacion: List[List[str]], distancias: List[int], k: int = 3, rng: random.Random = None) -> List[str]:
    """
    Selecciona un individuo mediante torneo de tamaño k.
    Gana el de menor distancia (mayor aptitud).
    """
    if rng is None:
        rng = random

    indices_seleccionados = rng.sample(range(len(poblacion)), k)
    mejor_idx = min(indices_seleccionados, key=lambda idx: distancias[idx])
    return poblacion[mejor_idx].copy()


def cruzamiento_ox(padre1: List[str], padre2: List[str], rng: random.Random = None) -> List[str]:
    """
    Cruzamiento por Orden (Order Crossover - OX):
    Copia un subsegmento continuo de padre1 y rellena los huecos restantes
    con los elementos de padre2 respetando su orden de aparición relativo.
    """
    if rng is None:
        rng = random

    n = len(padre1)
    i, j = sorted(rng.sample(range(n), 2))

    # Segmento heredado de padre1
    hijo = [None] * n
    hijo[i:j + 1] = padre1[i:j + 1]

    segmento_set = set(padre1[i:j + 1])
    elementos_restantes = [gen for gen in padre2 if gen not in segmento_set]

    # Rellenar los espacios vacíos
    idx_restante = 0
    for pos in range(n):
        if hijo[pos] is None:
            hijo[pos] = elementos_restantes[idx_restante]
            idx_restante += 1

    return hijo


def mutacion_swap(individuo: List[str], rng: random.Random = None) -> List[str]:
    """
    Mutación por intercambio (Swap Mutation):
    Intercambia la posición de dos genes elegidos al azar.
    """
    if rng is None:
        rng = random

    n = len(individuo)
    i, j = rng.sample(range(n), 2)
    mutado = individuo.copy()
    mutado[i], mutado[j] = mutado[j], mutado[i]
    return mutado


def ejecutar_algoritmo_genetico_una_semilla(
    semilla: int,
    tam_poblacion: int = 10,
    generaciones: int = 30,
    k_torneo: int = 3,
    prob_cruzamiento: float = 0.85,
    prob_mutacion: float = 0.20
) -> Dict[str, Any]:
    """
    Ejecuta el ciclo evolutivo de 30 generaciones para una semilla fija dada.
    """
    rng = random.Random(semilla)
    poblacion = generar_poblacion_inicial(tam_poblacion, rng)

    # Evaluar población inicial
    distancias = [calcular_distancia_cromosoma(ind) for ind in poblacion]
    mejor_dist_inicial = min(distancias)
    mejor_ind_inicial = poblacion[distancias.index(mejor_dist_inicial)].copy()

    historial_generaciones = []
    poblacion_inicial_guardada = [ind.copy() for ind in poblacion]

    mejor_global_dist = mejor_dist_inicial
    mejor_global_ind = mejor_ind_inicial.copy()

    for gen in range(1, generaciones + 1):
        # Encontrar el mejor de la generación actual
        min_dist_gen = min(distancias)
        idx_mejor_gen = distancias.index(min_dist_gen)
        mejor_ind_gen = poblacion[idx_mejor_gen]

        if min_dist_gen < mejor_global_dist:
            mejor_global_dist = min_dist_gen
            mejor_global_ind = mejor_ind_gen.copy()

        historial_generaciones.append({
            'generacion': gen,
            'mejor_distancia': min_dist_gen,
            'mejor_individuo': mejor_ind_gen.copy()
        })

        if gen == generaciones:
            break

        # Construir nueva población con Elitismo (1 individuo)
        nueva_poblacion = [mejor_ind_gen.copy()]

        # Generar los 9 individuos restantes
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, distancias, k_torneo, rng)
            padre2 = seleccion_torneo(poblacion, distancias, k_torneo, rng)

            if rng.random() < prob_cruzamiento:
                hijo = cruzamiento_ox(padre1, padre2, rng)
            else:
                hijo = padre1.copy()

            if rng.random() < prob_mutacion:
                hijo = mutacion_swap(hijo, rng)

            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion
        distancias = [calcular_distancia_cromosoma(ind) for ind in poblacion]

    return {
        'semilla': semilla,
        'poblacion_inicial': poblacion_inicial_guardada,
        'mejor_dist_inicial': mejor_dist_inicial,
        'mejor_ind_inicial': mejor_ind_inicial,
        'mejor_dist_final': mejor_global_dist,
        'mejor_ind_final': mejor_global_ind,
        'mejor_cromosoma_str': "–".join(mejor_global_ind),
        'historial_generaciones': historial_generaciones
    }


def ejecutar_5_corridas(semillas: List[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ejecuta el Algoritmo Genético sobre las 5 semillas registradas
    y recopila los resultados comparativos y la mejor solución global.
    """
    if semillas is None:
        semillas = SEMILLAS_OFICIALES

    resultados_corridas = []
    mejor_absoluto = None

    for i, sem in enumerate(semillas, 1):
        res = ejecutar_algoritmo_genetico_una_semilla(sem)
        res['ejecucion'] = i
        resultados_corridas.append(res)

        if mejor_absoluto is None or res['mejor_dist_final'] < mejor_absoluto['mejor_dist_final']:
            mejor_absoluto = res

    return resultados_corridas, mejor_absoluto


def desglosar_ruta_fisica(cromosoma: List[str]) -> List[Dict[str, Any]]:
    """
    Desglosa paso a paso los tramos físicos reales autorizados en el campus
    para un orden de entrega dado, indicando pasillos intermedios.
    """
    orden_completo = ['A'] + cromosoma + ['A']
    tramos = []
    acumulado = 0

    for i in range(len(orden_completo) - 1):
        u = orden_completo[i]
        v = orden_completo[i + 1]
        dist_tramo = MATRIZ_DISTANCIAS_MINIMAS[u][v]
        camino_fisico = RUTAS_FISICAS_PARES.get((u, v), [u, v])
        acumulado += dist_tramo

        es_retorno = (i == len(orden_completo) - 2)
        es_salida = (i == 0)

        if es_salida:
            tipo = f"Salida {u} → {v}"
            accion = f"Entrega en {v} ({NODOS_CAMPUS[v]})"
        elif es_retorno:
            tipo = f"Regreso {u} → {v}"
            accion = f"Retorno al almacén {v} ({NODOS_CAMPUS[v]})"
        else:
            tipo = f"Traslado {u} → {v}"
            accion = f"Entrega en {v} ({NODOS_CAMPUS[v]})"

        camino_str = " → ".join(camino_fisico)
        tramos.append({
            'paso': i + 1,
            'tipo': tipo,
            'camino_fisico': camino_str,
            'distancia_tramo': dist_tramo,
            'acumulado': acumulado,
            'accion': accion
        })

    return tramos


def verificar_120_ordenes() -> Dict[str, Any]:
    """
    Evalúa exhaustivamente las 120 (5!) permutaciones posibles de los 5 destinos
    para obtener el óptimo global teórico, el peor recorrido y la distribución.
    """
    todas_permutaciones = list(itertools.permutations(DESTINOS))
    resultados = []

    for perm in todas_permutaciones:
        dist = calcular_distancia_cromosoma(list(perm))
        resultados.append((dist, list(perm)))

    resultados.sort(key=lambda x: x[0])
    min_dist = resultados[0][0]
    max_dist = resultados[-1][0]
    optimos = [p for d, p in resultados if d == min_dist]
    peores = [p for d, p in resultados if d == max_dist]

    distancias = [d for d, _ in resultados]
    promedio = sum(distancias) / len(distancias)

    return {
        'total_evaluados': len(todas_permutaciones),
        'min_dist': min_dist,
        'max_dist': max_dist,
        'promedio_dist': promedio,
        'optimos': optimos,
        'optimos_str': [f"{'–'.join(p)} ({min_dist} m)" for p in optimos],
        'peor_ejemplo_str': f"{'–'.join(peores[0])} ({max_dist} m)"
    }


def obtener_ejemplo_ox_documentado() -> Dict[str, Any]:
    """Devuelve el ejemplo pedagógico de Cruzamiento por Orden (OX) documentado en la plantilla."""
    padre1 = ['B', 'D', 'F', 'E', 'C']
    padre2 = ['C', 'E', 'B', 'D', 'F']
    # Segmento índices 1 a 3: ['D', 'F', 'E']
    hijo = ['C', 'D', 'F', 'E', 'B']
    return {
        'padre1': padre1,
        'padre2': padre2,
        'segmento_indices': (1, 3),
        'segmento_valores': ['D', 'F', 'E'],
        'restantes_padre2': ['C', 'B'],
        'hijo': hijo
    }


def obtener_ejemplo_swap_documentado() -> Dict[str, Any]:
    """Devuelve el ejemplo pedagógico de Mutación por Intercambio (Swap) documentado en la plantilla."""
    antes = ['C', 'D', 'F', 'E', 'B']
    # Intercambio de pos 1 ('D') y 3 ('E')
    despues = ['C', 'E', 'F', 'D', 'B']
    return {
        'antes': antes,
        'posiciones': (1, 3),
        'genes': ('D', 'E'),
        'despues': despues
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    corridas, mejor = ejecutar_5_corridas()
    print("=== RESULTADOS 5 EJECUCIONES ALGORITMO GENÉTICO ===")
    print(f"{'Ejecución':<10} | {'Semilla':<8} | {'D. Inicial':<12} | {'D. Final':<10} | {'Mejor Cromosoma'}")
    print("-" * 65)
    for c in corridas:
        print(f"{c['ejecucion']:<10} | {c['semilla']:<8} | {c['mejor_dist_inicial']} m{'':<8} | {c['mejor_dist_final']} m{'':<6} | {c['mejor_cromosoma_str']}")

    print("\n=== VERIFICACIÓN EXHAUSTIVA DE LAS 120 PERMUTACIONES ===")
    verif = verificar_120_ordenes()
    print(f"Total evaluados: {verif['total_evaluados']}")
    print(f"Distancia mínima exacta (Óptimo Global): {verif['min_dist']} m")
    print(f"Óptimos encontrados: {', '.join(verif['optimos_str'])}")
    print(f"Distancia máxima (Peor caso): {verif['peor_ejemplo_str']}")
    print(f"Distancia promedio: {verif['promedio_dist']:.1f} m")

    print("\n=== DESGLOSE DE RUTA FÍSICA MEJOR INDIVIDUO ===")
    tramos = desglosar_ruta_fisica(mejor['mejor_ind_final'])
    for t in tramos:
        print(f"Paso {t['paso']}: {t['tipo']:<18} | Camino real: {t['camino_fisico']:<18} | {t['distancia_tramo']} m (Acum: {t['acumulado']} m) | {t['accion']}")
