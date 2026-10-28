# Mini Sistema de Algoritmos de Búsqueda y Optimización (Campus IA)

Sistema interactivo desarrollado en **Python (CustomTkinter + Matplotlib + NetworkX)** con estética industrial (**Guía de Estilos ORBIT**) para representar, simular y analizar tres problemas fundamentales de Inteligencia Artificial aplicados a la logística de reparto de un robot autónomo en un campus universitario:

1. **Práctica 1 (Búsqueda no informada - BFS):** Localizar una ruta entre el *Almacén Central (`A`)* y el *Laboratorio de Cómputo (`F`)* minimizando el número de conexiones (saltos).
2. **Práctica 2 (Búsqueda informada y heurística - $A^*$):** Localizar la ruta de menor distancia en metros entre `A` y `F` guiada por una función heurística admisible $f(n) = g(n) + h(n)$.
3. **Práctica 3 (Computación evolutiva - Algoritmo Genético):** Planificar el orden óptimo de entrega para visitar los 5 destinos (`B, C, D, E, F`) con salida y retorno en `A` (problema TSP) utilizando la matriz de distancias mínimas precalculada con Dijkstra.

---

## 🎨 Sistema de Diseño y Paleta Visual (ORBIT)

El diseño de la aplicación sigue la estética **industrial neutra con acento amarillo** documentada en `style_guide_palette.md`:

- **Lienzo y Fondos (`#181818`, `#252526`):** Fondos oscuros en escala neutra pura (antracita y acero) para máximo confort visual.
- **Superficies y Paneles (`#252526`, `#2D2D30`):** Contenedores de datos, tarjetas de métricas y tablas Treeview.
- **Bordes Estructurales (`#333333`, `#444444`):** Separadores y contornos sutiles de alta definición.
- **Acento Amarillo Industrial (`#F2E205` / `#F2EA7E`):** Reservado para botones de acción principal (`▶ Ejecutar`), pestañas activas, títulos y nodos destacados.
- **Contraste Accesible (`#12110D`):** Color oscuro de alto contraste (WCAG AAA) para textos sobre superficies amarillas.
- **Tipografía Neutra (`#F5F5F5`, `#A0A0A0`, `#707070`):** Texto principal, secundario y etiquetas libres de tintes cálidos.

---

## 🚀 Requisitos e Instalación

### 1. Requisitos Previos
- Python versión **3.10 o superior** (compatible con 3.11, 3.12, 3.13 y 3.14).

### 2. Instalación de Dependencias
Ejecuta el siguiente comando en la terminal desde el directorio del proyecto:
```bash
pip install -r requirements.txt
```
*(Librerías principales instaladas: `customtkinter`, `matplotlib`, `networkx`)*

---

## 💻 Comando para Ejecutar el Sistema

Para iniciar la aplicación gráfica completa:

```bash
python main.py
```

*(O alternativamente: `python src/main_gui.py`)*

---

## 🧪 Ejecución de Pruebas Automatizadas

El proyecto incluye una suite de pruebas unitarias que verifica la corrección de Dijkstra, BFS, $A^*$ y el Algoritmo Genético contra los valores documentados en el portafolio:

```bash
python -m unittest tests.test_algoritmos
```

### Ejecución de Módulos Individuales en Consola (CLI):
- **Práctica 1 (BFS):**
  ```bash
  python -m src.practica1_bfs
  ```
- **Práctica 2 ($A^*$):**
  ```bash
  python -m src.practica2_astar
  ```
- **Práctica 3 (Algoritmo Genético):**
  ```bash
  python -m src.practica3_genetico
  ```

---

## 📁 Estructura del Repositorio

```text
Portafolio/
├── .gitignore               # Exclusión de archivos compilados, caché y temporales
├── requirements.txt         # Dependencias externas del proyecto
├── main.py                  # Punto de entrada y lanzador principal
├── plan.md                  # Plan de requerimientos original del sistema
├── Plantilla_Portafolio_IA_Practicas_1_2_3.md # Reporte académico extenso
├── style_guide_palette.md   # Especificación de la paleta de colores ORBIT
├── README.md                # Documentación del sistema
├── src/
│   ├── __init__.py
│   ├── datos_campus.py      # Catálogo de áreas, aristas, heurística h(n) y Dijkstra
│   ├── practica1_bfs.py     # Búsqueda en anchura con traza FIFO y métricas
│   ├── practica2_astar.py   # Búsqueda A* con desempate alfabético y relajación
│   ├── practica3_genetico.py# TSP con torneo (k=3), OX, Swap, elitismo y 5 semillas
│   └── main_gui.py          # GUI en CustomTkinter con paleta ORBIT y 4 pestañas
└── tests/
    ├── __init__.py
    └── test_algoritmos.py   # Pruebas unitarias de optimalidad y verificación
```

---

## 🖥️ Descripción de las 4 Vistas del Sistema

### Vista 1: Preparación y Grafo del Campus
- **Panel Izquierdo:** Catálogo de las 6 áreas universitarias ($A$ a $F$), lista de adyacencia ordenada alfabéticamente con pesos en metros ($w > 0$) y marco de reglas operativas.
- **Panel Derecho:** Renderizado interactivo del grafo del campus (NetworkX + Matplotlib) con nodos en amarillo industrial, pesos de pasillos y botones para resaltar en vivo las rutas de BFS, $A^*$ o restablecer la vista.

### Vista 2: Práctica 1 - Búsqueda Ciega (BFS)
- **Barra de Control:** Origen fijo `A` (Almacén) y meta `F` (Cómputo) con botón de ejecución.
- **Tabla de Seguimiento:** `Treeview` de 5 columnas (*Paso, Nodo procesado, Cola final, Nodos descubiertos, Padres / Observaciones*).
- **Tarjetas de Resultados:** Ruta encontrada (`A → B → D → F`), 3 conexiones, distancia total de `210 m` y 5 nodos expandidos (`A, B, C, D, E`).

### Vista 3: Práctica 2 - Búsqueda Heurística ($A^*$)
- **Barra de Control:** Resumen visible de la heurística admisible $h(n)$ hacia $F$ y botón de ejecución.
- **Tabla de Iteraciones:** `Treeview` de 6 columnas (*Iteración, Nodo seleccionado, g(n), h(n), f(n), Cambios en frontera*), reflejando el desempate alfabético y la relajación de la arista hacia $F$ (`160 m < 170 m`).
- **Resultados y Comparativa:** Tarjetas de métricas (`A → C → E → F`, 3 conexiones, `160 m`, 5 expandidos) y tabla comparativa lado a lado frente a BFS resaltando el ahorro del **23.8% (50 m)**.

### Vista 4: Práctica 3 - Algoritmo Genético
- **Panel Superior:** Parámetros configurados ($N=10$, $G=30$, $k=3$, $P_c=0.85$, $P_m=0.20$, Elitismo=1, Semillas: 42, 101, 2026, 777, 999).
- **Sub-panel Izquierdo:** Matriz de distancias mínimas $6 \times 6$ precalculada con Dijkstra y visor pedagógico de operadores (población inicial, ejemplo paso a paso de OX y de Swap).
- **Sub-panel Central:** Tabla oficial de las 5 ejecuciones, desglose físico tramo a tramo del óptimo global (`360 m`) y verificación exhaustiva contra las 120 permutaciones posibles ($5!$).
- **Sub-panel Derecho:** Contenedor de Matplotlib incrustado con las 5 curvas generacionales de convergencia.
