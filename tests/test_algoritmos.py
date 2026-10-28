"""
Suite de Pruebas Automatizadas para el Mini Sistema de Búsqueda
Verifica la corrección de Dijkstra, BFS, A* y Algoritmo Genético,
contrastando contra los resultados teóricos documentados en el portafolio.
"""

import unittest
from src.datos_campus import (
    GRAFO_CAMPUS,
    HEURISTICA_F,
    MATRIZ_DISTANCIAS_MINIMAS,
    RUTAS_FISICAS_PARES
)
from src.practica1_bfs import ejecutar_bfs
from src.practica2_astar import ejecutar_astar
from src.practica3_genetico import (
    ejecutar_algoritmo_genetico_una_semilla,
    verificar_120_ordenes,
    desglosar_ruta_fisica
)


class TestAlgoritmosBusqueda(unittest.TestCase):

    def test_datos_campus_dijkstra(self):
        """Verifica la matriz 6x6 de distancias mínimas y heurísticas admisibles."""
        # A hacia F debe ser 160 m
        self.assertEqual(MATRIZ_DISTANCIAS_MINIMAS['A']['F'], 160)
        # Camino A a F: A -> C -> E -> F
        self.assertEqual(RUTAS_FISICAS_PARES[('A', 'F')], ['A', 'C', 'E', 'F'])
        # Admisibilidad de h(n) <= h*(n)
        for nodo, h_val in HEURISTICA_F.items():
            dist_real = MATRIZ_DISTANCIAS_MINIMAS[nodo]['F']
            self.assertLessEqual(h_val, dist_real, f"h({nodo}) no es admisible")

    def test_practica1_bfs(self):
        """Verifica los resultados exactos de la búsqueda en anchura."""
        res = ejecutar_bfs('A', 'F')
        self.assertTrue(res['encontrado'])
        self.assertEqual(res['ruta'], ['A', 'B', 'D', 'F'])
        self.assertEqual(res['conexiones'], 3)
        self.assertEqual(res['distancia_total'], 210)
        self.assertEqual(res['nodos_expandidos'], ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(res['total_expandidos'], 5)
        # La tabla debe tener exactamente 7 filas (Paso Inicial + 6 pasos)
        self.assertEqual(len(res['historial_pasos']), 7)

    def test_practica2_astar(self):
        """Verifica los resultados exactos de A* con desempate alfabético y relajación de aristas."""
        res = ejecutar_astar('A', 'F')
        self.assertTrue(res['encontrado'])
        self.assertEqual(res['ruta'], ['A', 'C', 'E', 'F'])
        self.assertEqual(res['conexiones'], 3)
        self.assertEqual(res['distancia_total'], 160)
        self.assertEqual(res['nodos_expandidos'], ['A', 'C', 'D', 'B', 'E'])
        self.assertEqual(res['total_expandidos'], 5)
        # La tabla debe tener 6 iteraciones
        self.assertEqual(len(res['historial_iteraciones']), 6)

    def test_practica3_genetico_optimo_global(self):
        """Verifica las 120 permutaciones posibles y el óptimo de 360 m."""
        verif = verificar_120_ordenes()
        self.assertEqual(verif['total_evaluados'], 120)
        self.assertEqual(verif['min_dist'], 360)
        self.assertEqual(verif['max_dist'], 640)
        self.assertEqual(verif['promedio_dist'], 512.0)

        # Probar que la semilla 42 converge
        res_sem42 = ejecutar_algoritmo_genetico_una_semilla(42)
        self.assertEqual(res_sem42['mejor_dist_final'], 360)

        # Probar desglose físico
        tramos = desglosar_ruta_fisica(res_sem42['mejor_ind_final'])
        self.assertEqual(len(tramos), 6)
        self.assertEqual(tramos[-1]['acumulado'], 360)


if __name__ == "__main__":
    unittest.main()
