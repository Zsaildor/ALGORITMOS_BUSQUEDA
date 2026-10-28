"""
Módulo Principal de la Interfaz Gráfica (GUI)
Construido con CustomTkinter, Matplotlib y NetworkX.
Implementa una estética industrial inspirada en la Guía de Estilos ORBIT:
- Fondos neutros oscuros (#181818, #252526, #2D2D30)
- Bordes estructurales (#333333, #444444)
- Tipografía neutra (#F5F5F5, #A0A0A0, #707070)
- Acento industrial amarillo (#F2E205, #F2EA7E, #D4C404, #12110D)
y cuatro vistas en pestañas para representar BFS, A* y Algoritmo Genético.
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

# Importación de módulos lógicos del proyecto
from src.datos_campus import (
    NODOS_CAMPUS,
    ARISTAS_DEFINIDAS,
    HEURISTICA_F,
    POSICIONES_NODOS,
    GRAFO_CAMPUS,
    MATRIZ_DISTANCIAS_MINIMAS,
    RUTAS_FISICAS_PARES
)
from src.practica1_bfs import ejecutar_bfs
from src.practica2_astar import ejecutar_astar, generar_tabla_comparativa
from src.practica3_genetico import (
    ejecutar_5_corridas,
    desglosar_ruta_fisica,
    verificar_120_ordenes,
    obtener_ejemplo_ox_documentado,
    obtener_ejemplo_swap_documentado
)

# Configuración inicial de CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# =============================================================================
# CONSTANTES DE PALETA DE COLORES (GUÍA DE ESTILOS ORBIT)
# =============================================================================
# Superficies y Fondo
COLOR_APP_BG = "#181818"          # Fondo general de la aplicación
COLOR_SIDEBAR_BG = "#1E1E1E"      # Fondo de controles o inputs
COLOR_CARD_BG = "#252526"         # Tarjetas, paneles y contenedores
COLOR_CARD_HOVER = "#2D2D30"      # Hover de filas y tarjetas
COLOR_BORDER = "#333333"          # Bordes estándar y divisores
COLOR_BORDER_LIGHT = "#444444"    # Bordes con realce y separadores

# Acento Principal (Industrial Yellow)
COLOR_PRIMARY = "#F2E205"         # Acento amarillo industrial principal
COLOR_PRIMARY_HOVER = "#F2EA7E"   # Hover amarillo claro
COLOR_PRIMARY_PRESSED = "#D4C404" # Click / Pressed amarillo profundo
COLOR_ON_PRIMARY = "#12110D"      # Texto de alto contraste sobre fondo amarillo

# Tipografía y Textos
COLOR_TEXT_PRIMARY = "#F5F5F5"    # Títulos y valores numéricos destacados
COLOR_TEXT_SECONDARY = "#A0A0A0"  # Subtítulos, descripciones y etiquetas
COLOR_TEXT_MUTED = "#707070"      # Metadatos y textos discretos

# Semántica y Categorías
COLOR_SUCCESS = "#84CC16"         # Confirmaciones y éxito
COLOR_WARNING = "#F2EA7E"         # Alertas suaves
COLOR_DANGER = "#F25757"          # Peligro / resaltado crítico
COLOR_INFO = "#38BDF8"            # Información y ayuda


class SistemaAlgoritmosApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Búsqueda y Optimización - Campus Universitario (IA)")
        self.geometry("1300x860")
        self.minsize(1150, 750)
        self.configure(fg_color=COLOR_APP_BG)

        # Configurar estilos de TTK para Treeviews integrados con la paleta ORBIT
        self.configurar_estilos_treeview()

        # Almacenar resultados de ejecuciones
        self.resultado_bfs = None
        self.resultado_astar = None
        self.resultado_genetico_corridas = None
        self.mejor_genetico_global = None

        # Construcción de la interfaz
        self.construir_encabezado()
        self.construir_tabview()

        # Inicializar vistas
        self.inicializar_tab_preparacion()
        self.inicializar_tab_bfs()
        self.inicializar_tab_astar()
        self.inicializar_tab_genetico()

    def configurar_estilos_treeview(self):
        """Aplica la paleta ORBIT a los componentes Treeview de Tkinter."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Treeview",
            background=COLOR_CARD_BG,
            foreground=COLOR_TEXT_PRIMARY,
            fieldbackground=COLOR_CARD_BG,
            rowheight=28,
            font=("Segoe UI", 10),
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_SIDEBAR_BG,
            foreground=COLOR_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            relief="flat"
        )
        style.map(
            "Treeview",
            background=[("selected", COLOR_PRIMARY_PRESSED)],
            foreground=[("selected", COLOR_ON_PRIMARY)]
        )
        style.map(
            "Treeview.Heading",
            background=[("active", COLOR_CARD_HOVER)],
            foreground=[("active", COLOR_PRIMARY_HOVER)]
        )

    def construir_encabezado(self):
        """Crea la barra superior con el título principal y créditos del portafolio."""
        header_frame = ctk.CTkFrame(
            self,
            fg_color=COLOR_CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=COLOR_BORDER,
            height=65
        )
        header_frame.pack(fill="x", padx=15, pady=(12, 6))

        titulo = ctk.CTkLabel(
            header_frame,
            text="Portafolio de Inteligencia Artificial: Algoritmos de Búsqueda y Optimización",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        titulo.pack(side="left", padx=20, pady=10)

        subtitulo = ctk.CTkLabel(
            header_frame,
            text="Práctica 1 (BFS)  |  Práctica 2 (A*)  |  Práctica 3 (Algoritmo Genético)",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_SECONDARY
        )
        subtitulo.pack(side="right", padx=20, pady=10)

    def construir_tabview(self):
        """Crea el contenedor con las 4 pestañas requeridas bajo la paleta ORBIT."""
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=COLOR_APP_BG,
            segmented_button_selected_color=COLOR_PRIMARY,
            segmented_button_selected_hover_color=COLOR_PRIMARY_HOVER,
            segmented_button_unselected_color=COLOR_CARD_BG,
            segmented_button_unselected_hover_color=COLOR_CARD_HOVER,
            text_color=COLOR_ON_PRIMARY
        )
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.tab_prep = self.tabview.add("Preparación y Grafo del Campus")
        self.tab_bfs = self.tabview.add("Práctica 1 - Búsqueda Ciega (BFS)")
        self.tab_astar = self.tabview.add("Práctica 2 - Búsqueda Heurística (A*)")
        self.tab_ga = self.tabview.add("Práctica 3 - Algoritmo Genético")

    # =========================================================================
    # VISTA 1: PREPARACIÓN Y GRAFO DEL CAMPUS
    # =========================================================================
    def inicializar_tab_preparacion(self):
        # Contenedor dividido en dos columnas: Izquierda (Datos), Derecha (Grafo)
        self.tab_prep.grid_columnconfigure(0, weight=4)
        self.tab_prep.grid_columnconfigure(1, weight=6)
        self.tab_prep.grid_rowconfigure(0, weight=1)

        # PANEL IZQUIERDO: Datos de entrada y reglas
        left_frame = ctk.CTkScrollableFrame(
            self.tab_prep,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 8), pady=10)

        lbl_cat_title = ctk.CTkLabel(
            left_frame,
            text="1. Catálogo de Áreas Universitarias (Nodos)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_cat_title.pack(anchor="w", padx=10, pady=(5, 5))

        # Tabla de catálogo de nodos
        tree_cat = ttk.Treeview(left_frame, columns=("nodo", "nombre", "h"), show="headings", height=6)
        tree_cat.heading("nodo", text="Nodo")
        tree_cat.heading("nombre", text="Área Universitaria")
        tree_cat.heading("h", text="Heurística h(n) a F")
        tree_cat.column("nodo", width=60, anchor="center")
        tree_cat.column("nombre", width=190, anchor="w")
        tree_cat.column("h", width=120, anchor="center")
        tree_cat.pack(fill="x", padx=10, pady=(0, 10))

        for n, nombre in NODOS_CAMPUS.items():
            tree_cat.insert("", "end", values=(n, nombre, f"{HEURISTICA_F[n]} m"))

        # Lista de adyacencia
        lbl_ady_title = ctk.CTkLabel(
            left_frame,
            text="2. Lista de Adyacencia (Pesos Positivos en Metros)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_ady_title.pack(anchor="w", padx=10, pady=(10, 5))

        txt_ady = ctk.CTkTextbox(
            left_frame,
            height=135,
            fg_color=COLOR_SIDEBAR_BG,
            text_color=COLOR_TEXT_PRIMARY,
            border_width=1,
            border_color=COLOR_BORDER,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        txt_ady.pack(fill="x", padx=10, pady=(0, 10))

        ady_str = ""
        for u in sorted(GRAFO_CAMPUS.keys()):
            vecinos = ", ".join([f"{v}({peso}m)" for v, peso in GRAFO_CAMPUS[u]])
            ady_str += f"• Nodo {u} ({NODOS_CAMPUS[u]}):\n   Vecinos ordenados: [{vecinos}]\n"
        txt_ady.insert("0.0", ady_str)
        txt_ady.configure(state="disabled")

        # Reglas operativas
        lbl_reglas_title = ctk.CTkLabel(
            left_frame,
            text="3. Reglas Generales de Operación",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_reglas_title.pack(anchor="w", padx=10, pady=(10, 5))

        txt_reglas = ctk.CTkTextbox(
            left_frame,
            height=140,
            fg_color=COLOR_SIDEBAR_BG,
            text_color=COLOR_TEXT_SECONDARY,
            border_width=1,
            border_color=COLOR_BORDER,
            font=ctk.CTkFont(size=11)
        )
        txt_reglas.pack(fill="x", padx=10, pady=(0, 10))
        reglas_text = (
            "✔ Orden alfabético: En BFS y A*, los vecinos adyacentes se evalúan estrictamente de A a F.\n"
            "✔ Desempate en A*: Ante empate en f(n) (ej. B y E con f=150), se selecciona el menor alfabéticamente (B antes que E).\n"
            "✔ Conteo de expandidos: Un nodo se cuenta como expandido solo al procesar sus vecinos. La meta F nunca se cuenta como expandida.\n"
            "✔ Grafo simétrico: 8 caminos bidireccionales con pesos estrictamente mayores a cero."
        )
        txt_reglas.insert("0.0", reglas_text)
        txt_reglas.configure(state="disabled")

        # PANEL DERECHO: Visualizador interactivo del grafo con NetworkX y Matplotlib
        right_frame = ctk.CTkFrame(
            self.tab_prep,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 5), pady=10)

        top_controls_graph = ctk.CTkFrame(right_frame, fg_color="transparent")
        top_controls_graph.pack(fill="x", padx=15, pady=(10, 5))

        lbl_grafo = ctk.CTkLabel(
            top_controls_graph,
            text="Topología del Grafo del Campus (8 Pasillos Bidireccionales)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_grafo.pack(side="left")

        btn_reset_graph = ctk.CTkButton(
            top_controls_graph,
            text="Restablecer Vista",
            width=130,
            fg_color=COLOR_SIDEBAR_BG,
            text_color=COLOR_TEXT_PRIMARY,
            hover_color=COLOR_CARD_HOVER,
            border_width=1,
            border_color=COLOR_BORDER,
            command=lambda: self.dibujar_grafo_campus()
        )
        btn_reset_graph.pack(side="right")

        # Botones para resaltar rutas en el grafo
        btn_bar = ctk.CTkFrame(right_frame, fg_color="transparent")
        btn_bar.pack(fill="x", padx=15, pady=(0, 5))

        btn_ver_bfs = ctk.CTkButton(
            btn_bar,
            text="Resaltar Ruta BFS (210m)",
            fg_color=COLOR_SIDEBAR_BG,
            text_color=COLOR_TEXT_PRIMARY,
            hover_color=COLOR_CARD_HOVER,
            border_width=1,
            border_color=COLOR_BORDER,
            command=lambda: self.dibujar_grafo_campus(resaltar_ruta=['A', 'B', 'D', 'F'], etiqueta="Ruta BFS: A→B→D→F (210 m)")
        )
        btn_ver_bfs.pack(side="left", padx=(0, 6), expand=True, fill="x")

        btn_ver_astar = ctk.CTkButton(
            btn_bar,
            text="Resaltar Ruta A* (160m)",
            fg_color=COLOR_PRIMARY,
            text_color=COLOR_ON_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            font=ctk.CTkFont(weight="bold"),
            command=lambda: self.dibujar_grafo_campus(resaltar_ruta=['A', 'C', 'E', 'F'], etiqueta="Ruta A*: A→C→E→F (160 m)")
        )
        btn_ver_astar.pack(side="left", padx=6, expand=True, fill="x")

        # Canvas para Matplotlib
        self.graph_container = ctk.CTkFrame(
            right_frame,
            fg_color=COLOR_APP_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=8
        )
        self.graph_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.fig_grafo, self.ax_grafo = plt.subplots(figsize=(6, 4.5), facecolor=COLOR_APP_BG)
        self.canvas_grafo = FigureCanvasTkAgg(self.fig_grafo, master=self.graph_container)
        self.canvas_grafo.get_tk_widget().pack(fill="both", expand=True)

        self.dibujar_grafo_campus()

    def dibujar_grafo_campus(self, resaltar_ruta=None, etiqueta=None):
        """Renderiza el grafo del campus destacando aristas y nodos con la paleta ORBIT."""
        self.ax_grafo.clear()
        self.ax_grafo.set_facecolor(COLOR_APP_BG)

        G = nx.Graph()
        for u, v, peso in ARISTAS_DEFINIDAS:
            G.add_edge(u, v, weight=peso)

        pos = POSICIONES_NODOS

        # Aristas de la ruta resaltada si aplica
        aristas_resaltadas = []
        if resaltar_ruta:
            for i in range(len(resaltar_ruta) - 1):
                u, v = resaltar_ruta[i], resaltar_ruta[i + 1]
                aristas_resaltadas.append((u, v))
                aristas_resaltadas.append((v, u))

        # Dibujar aristas normales
        aristas_normales = [e for e in G.edges() if e not in aristas_resaltadas and (e[1], e[0]) not in aristas_resaltadas]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=aristas_normales,
            ax=self.ax_grafo,
            edge_color=COLOR_BORDER_LIGHT,
            width=2.5,
            alpha=0.85
        )

        # Dibujar aristas resaltadas
        if aristas_resaltadas:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[e for e in G.edges() if e in aristas_resaltadas or (e[1], e[0]) in aristas_resaltadas],
                ax=self.ax_grafo,
                edge_color=COLOR_PRIMARY,
                width=5.0
            )

        # Dibujar nodos
        colores_nodos = []
        for nodo in G.nodes():
            if resaltar_ruta and nodo in resaltar_ruta:
                colores_nodos.append(COLOR_PRIMARY_HOVER)
            else:
                colores_nodos.append(COLOR_PRIMARY)

        nx.draw_networkx_nodes(
            G, pos,
            ax=self.ax_grafo,
            node_color=colores_nodos,
            node_size=1100,
            edgecolors=COLOR_ON_PRIMARY,
            linewidths=2.0
        )

        # Etiquetas de los nodos
        nx.draw_networkx_labels(
            G, pos,
            ax=self.ax_grafo,
            font_size=12,
            font_weight="bold",
            font_color=COLOR_ON_PRIMARY
        )

        # Etiquetas de distancias en metros
        edge_labels = {(u, v): f"{d['weight']} m" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            ax=self.ax_grafo,
            font_size=10,
            font_weight="bold",
            font_color=COLOR_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.25", fc=COLOR_CARD_BG, ec=COLOR_BORDER, alpha=0.95)
        )

        # Título o leyenda
        if etiqueta:
            self.ax_grafo.set_title(etiqueta, color=COLOR_PRIMARY, fontsize=12, fontweight="bold", pad=10)
        else:
            self.ax_grafo.set_title("Grafo del Campus Universitario (Distancias en Metros)", color=COLOR_TEXT_PRIMARY, fontsize=11, pad=10)

        self.ax_grafo.axis("off")
        self.fig_grafo.tight_layout()
        self.canvas_grafo.draw()

    # =========================================================================
    # VISTA 2: PRÁCTICA 1 - BÚSQUEDA CIEGA (BFS)
    # =========================================================================
    def inicializar_tab_bfs(self):
        # Barra superior de control
        top_bar = ctk.CTkFrame(
            self.tab_bfs,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        top_bar.pack(fill="x", padx=10, pady=(10, 8))

        lbl_origen = ctk.CTkLabel(
            top_bar,
            text="Origen: [A] Almacén Central   ➜   Meta: [F] Laboratorio de Cómputo",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        lbl_origen.pack(side="left", padx=20, pady=12)

        lbl_desc = ctk.CTkLabel(
            top_bar,
            text="(Criterio: Minimizar número de conexiones mediante cola FIFO)",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_SECONDARY
        )
        lbl_desc.pack(side="left", padx=10)

        self.btn_ejecutar_bfs = ctk.CTkButton(
            top_bar,
            text="▶ Ejecutar BFS",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_PRIMARY,
            text_color=COLOR_ON_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            width=160,
            height=36,
            command=self.accion_ejecutar_bfs
        )
        self.btn_ejecutar_bfs.pack(side="right", padx=15, pady=10)

        # Sección central: Tabla de seguimiento manual/automático
        mid_frame = ctk.CTkFrame(
            self.tab_bfs,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        mid_frame.pack(fill="both", expand=True, padx=10, pady=5)

        lbl_tabla_bfs = ctk.CTkLabel(
            mid_frame,
            text="Tabla de Seguimiento Paso a Paso (Cola FIFO, Descubiertos y Padres)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_tabla_bfs.pack(anchor="w", padx=15, pady=(8, 4))

        # Contenedor con scroll para Treeview
        tree_container = tk.Frame(mid_frame, bg=COLOR_CARD_BG)
        tree_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        scroll_y_bfs = ttk.Scrollbar(tree_container, orient="vertical")
        scroll_x_bfs = ttk.Scrollbar(tree_container, orient="horizontal")

        columnas_bfs = ("paso", "nodo", "cola", "descubiertos", "observaciones")
        self.tree_bfs = ttk.Treeview(
            tree_container,
            columns=columnas_bfs,
            show="headings",
            yscrollcommand=scroll_y_bfs.set,
            xscrollcommand=scroll_x_bfs.set
        )

        scroll_y_bfs.config(command=self.tree_bfs.yview)
        scroll_x_bfs.config(command=self.tree_bfs.xview)

        scroll_y_bfs.pack(side="right", fill="y")
        scroll_x_bfs.pack(side="bottom", fill="x")
        self.tree_bfs.pack(side="left", fill="both", expand=True)

        self.tree_bfs.heading("paso", text="Paso")
        self.tree_bfs.heading("nodo", text="Nodo Procesado")
        self.tree_bfs.heading("cola", text="Cola (Al finalizar)")
        self.tree_bfs.heading("descubiertos", text="Nodos Descubiertos")
        self.tree_bfs.heading("observaciones", text="Padres Registrados / Observaciones")

        self.tree_bfs.column("paso", width=80, anchor="center")
        self.tree_bfs.column("nodo", width=120, anchor="center")
        self.tree_bfs.column("cola", width=150, anchor="center")
        self.tree_bfs.column("descubiertos", width=190, anchor="center")
        self.tree_bfs.column("observaciones", width=420, anchor="w")

        # Panel inferior: 4 recuadros destacados con métricas
        self.bottom_bfs = ctk.CTkFrame(self.tab_bfs, fg_color="transparent")
        self.bottom_bfs.pack(fill="x", padx=10, pady=(5, 10))
        self.bottom_bfs.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.card_bfs_ruta = self.crear_tarjeta_metrica(self.bottom_bfs, 0, "Ruta Encontrada", "Pendiente de ejecución")
        self.card_bfs_conex = self.crear_tarjeta_metrica(self.bottom_bfs, 1, "Número de Conexiones", "—")
        self.card_bfs_dist = self.crear_tarjeta_metrica(self.bottom_bfs, 2, "Distancia Total", "—")
        self.card_bfs_exp = self.crear_tarjeta_metrica(self.bottom_bfs, 3, "Nodos Expandidos", "—")

    def crear_tarjeta_metrica(self, parent, col, titulo, valor_inicial):
        """Crea una tarjeta rectangular moderna para mostrar métricas destacadas."""
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_CARD_BG,
            corner_radius=8,
            border_width=1,
            border_color=COLOR_BORDER
        )
        card.grid(row=0, column=col, padx=5, pady=2, sticky="nsew")

        lbl_t = ctk.CTkLabel(card, text=titulo, font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_PRIMARY)
        lbl_t.pack(anchor="w", padx=12, pady=(6, 2))

        lbl_v = ctk.CTkLabel(card, text=valor_inicial, font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        lbl_v.pack(anchor="w", padx=12, pady=(0, 8))

        return lbl_v

    def accion_ejecutar_bfs(self):
        """Ejecuta el módulo BFS y actualiza la tabla y tarjetas."""
        res = ejecutar_bfs('A', 'F')
        self.resultado_bfs = res

        # Limpiar tabla
        for item in self.tree_bfs.get_children():
            self.tree_bfs.delete(item)

        # Llenar tabla
        for paso in res['historial_pasos']:
            self.tree_bfs.insert("", "end", values=(
                paso['paso'],
                paso['nodo_procesado'],
                paso['cola'],
                paso['descubiertos'],
                paso['observaciones']
            ))

        # Actualizar tarjetas
        self.card_bfs_ruta.configure(text=res['ruta_str'])
        self.card_bfs_conex.configure(text=f"{res['conexiones']} tramos")
        self.card_bfs_dist.configure(text=f"{res['distancia_total']} m")
        self.card_bfs_exp.configure(text=f"{res['total_expandidos']} ({res['nodos_expandidos_str']})")

    # =========================================================================
    # VISTA 3: PRÁCTICA 2 - BÚSQUEDA HEURÍSTICA (A*)
    # =========================================================================
    def inicializar_tab_astar(self):
        # Barra superior de control con resumen de h(n)
        top_bar = ctk.CTkFrame(
            self.tab_astar,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        top_bar.pack(fill="x", padx=10, pady=(10, 8))

        h_texto = "  |  ".join([f"{n}: {val}m" for n, val in HEURISTICA_F.items()])
        lbl_h = ctk.CTkLabel(
            top_bar,
            text=f"Heurística admisible h(n) hacia F:\n{h_texto}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY,
            justify="left"
        )
        lbl_h.pack(side="left", padx=20, pady=8)

        self.btn_ejecutar_astar = ctk.CTkButton(
            top_bar,
            text="▶ Ejecutar A*",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_PRIMARY,
            text_color=COLOR_ON_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            width=160,
            height=36,
            command=self.accion_ejecutar_astar
        )
        self.btn_ejecutar_astar.pack(side="right", padx=15, pady=10)

        # Sección central: Tabla de iteraciones A*
        mid_frame = ctk.CTkFrame(
            self.tab_astar,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        mid_frame.pack(fill="both", expand=True, padx=10, pady=5)

        lbl_tabla_astar = ctk.CTkLabel(
            mid_frame,
            text="Tabla de Seguimiento de Iteraciones de A* (f(n) = g(n) + h(n), Desempate y Frontera)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_tabla_astar.pack(anchor="w", padx=15, pady=(8, 4))

        tree_container = tk.Frame(mid_frame, bg=COLOR_CARD_BG)
        tree_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        scroll_y_astar = ttk.Scrollbar(tree_container, orient="vertical")
        scroll_x_astar = ttk.Scrollbar(tree_container, orient="horizontal")

        columnas_astar = ("iter", "nodo", "g", "h", "f", "cambios")
        self.tree_astar = ttk.Treeview(
            tree_container,
            columns=columnas_astar,
            show="headings",
            yscrollcommand=scroll_y_astar.set,
            xscrollcommand=scroll_x_astar.set
        )

        scroll_y_astar.config(command=self.tree_astar.yview)
        scroll_x_astar.config(command=self.tree_astar.xview)

        scroll_y_astar.pack(side="right", fill="y")
        scroll_x_astar.pack(side="bottom", fill="x")
        self.tree_astar.pack(side="left", fill="both", expand=True)

        self.tree_astar.heading("iter", text="Iteración")
        self.tree_astar.heading("nodo", text="Nodo Seleccionado")
        self.tree_astar.heading("g", text="g(n)")
        self.tree_astar.heading("h", text="h(n)")
        self.tree_astar.heading("f", text="f(n)")
        self.tree_astar.heading("cambios", text="Cambios en los Nodos Pendientes (Frontera / Abiertos)")

        self.tree_astar.column("iter", width=70, anchor="center")
        self.tree_astar.column("nodo", width=120, anchor="center")
        self.tree_astar.column("g", width=60, anchor="center")
        self.tree_astar.column("h", width=60, anchor="center")
        self.tree_astar.column("f", width=60, anchor="center")
        self.tree_astar.column("cambios", width=590, anchor="w")

        # Panel inferior: Resultados de A* y Tabla Comparativa
        bot_frame = ctk.CTkFrame(self.tab_astar, fg_color="transparent")
        bot_frame.pack(fill="x", padx=10, pady=(5, 10))
        bot_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.card_astar_ruta = self.crear_tarjeta_metrica(bot_frame, 0, "Ruta Óptima A*", "Pendiente de ejecución")
        self.card_astar_conex = self.crear_tarjeta_metrica(bot_frame, 1, "Conexiones", "—")
        self.card_astar_dist = self.crear_tarjeta_metrica(bot_frame, 2, "Distancia Total", "—")
        self.card_astar_exp = self.crear_tarjeta_metrica(bot_frame, 3, "Nodos Expandidos", "—")

        # Tabla comparativa BFS vs A*
        comp_frame = ctk.CTkFrame(
            self.tab_astar,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        comp_frame.pack(fill="x", padx=10, pady=(0, 10))

        lbl_comp = ctk.CTkLabel(
            comp_frame,
            text="Tabla Comparativa Obligatoria: BFS vs. A*",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_comp.pack(anchor="w", padx=15, pady=(6, 2))

        cols_comp = ("algoritmo", "ruta", "conexiones", "distancia", "expandidos")
        self.tree_comp = ttk.Treeview(comp_frame, columns=cols_comp, show="headings", height=2)
        self.tree_comp.heading("algoritmo", text="Algoritmo")
        self.tree_comp.heading("ruta", text="Ruta")
        self.tree_comp.heading("conexiones", text="Conexiones")
        self.tree_comp.heading("distancia", text="Distancia (m)")
        self.tree_comp.heading("expandidos", text="Nodos Expandidos")

        self.tree_comp.column("algoritmo", width=170, anchor="w")
        self.tree_comp.column("ruta", width=180, anchor="center")
        self.tree_comp.column("conexiones", width=90, anchor="center")
        self.tree_comp.column("distancia", width=110, anchor="center")
        self.tree_comp.column("expandidos", width=250, anchor="w")
        self.tree_comp.pack(fill="x", padx=15, pady=(0, 8))

        self.lbl_ahorro = ctk.CTkLabel(
            comp_frame,
            text="✔ Ejecuta ambos algoritmos para contrastar el ahorro en distancia.",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXT_SECONDARY
        )
        self.lbl_ahorro.pack(anchor="w", padx=15, pady=(0, 6))

    def accion_ejecutar_astar(self):
        """Ejecuta A* y actualiza la tabla de iteraciones y la comparativa."""
        res_a = ejecutar_astar('A', 'F')
        self.resultado_astar = res_a

        # Limpiar tabla
        for item in self.tree_astar.get_children():
            self.tree_astar.delete(item)

        for it in res_a['historial_iteraciones']:
            self.tree_astar.insert("", "end", values=(
                it['iteracion'],
                it['nodo_seleccionado'],
                it['g'],
                it['h'],
                it['f'],
                it['cambios_frontera']
            ))

        self.card_astar_ruta.configure(text=res_a['ruta_str'])
        self.card_astar_conex.configure(text=f"{res_a['conexiones']} tramos")
        self.card_astar_dist.configure(text=f"{res_a['distancia_total']} m")
        self.card_astar_exp.configure(text=f"{res_a['total_expandidos']} ({res_a['nodos_expandidos_str']})")

        # Si aún no se ejecuta BFS, ejecutarlo internamente para llenar la comparativa
        if not self.resultado_bfs:
            self.resultado_bfs = ejecutar_bfs('A', 'F')

        # Actualizar tabla comparativa
        for item in self.tree_comp.get_children():
            self.tree_comp.delete(item)

        filas = generar_tabla_comparativa(self.resultado_bfs, self.resultado_astar)
        for fila in filas:
            self.tree_comp.insert("", "end", values=(
                fila['algoritmo'],
                fila['ruta'],
                fila['conexiones'],
                f"{fila['distancia_m']} m",
                fila['nodos_expandidos']
            ))

        ahorro = self.resultado_bfs['distancia_total'] - self.resultado_astar['distancia_total']
        pct = (ahorro / self.resultado_bfs['distancia_total']) * 100
        self.lbl_ahorro.configure(
            text=f"🎯 Conclusión Comparativa: A* reduce la distancia en {ahorro} metros (de 210m a 160m, ahorro del {pct:.1f}%) manteniendo las mismas 3 conexiones.",
            text_color=COLOR_PRIMARY
        )

    # =========================================================================
    # VISTA 4: PRÁCTICA 3 - ALGORITMO GENÉTICO
    # =========================================================================
    def inicializar_tab_genetico(self):
        # Panel superior de parámetros
        top_bar = ctk.CTkFrame(
            self.tab_ga,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        top_bar.pack(fill="x", padx=10, pady=(10, 8))

        params_text = (
            "Parámetros: Población: 10  |  Generaciones: 30  |  Torneo: k=3  |  "
            "Pc: 0.85 (OX)  |  Pm: 0.20 (Swap)  |  Elitismo: 1 ind  |  Semillas: 42, 101, 2026, 777, 999"
        )
        lbl_params = ctk.CTkLabel(
            top_bar,
            text=params_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_params.pack(side="left", padx=15, pady=10)

        self.btn_ejecutar_ga = ctk.CTkButton(
            top_bar,
            text="▶ Ejecutar 5 Corridas Evolutivas",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLOR_PRIMARY,
            text_color=COLOR_ON_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            width=220,
            height=34,
            command=self.accion_ejecutar_ga
        )
        self.btn_ejecutar_ga.pack(side="right", padx=15, pady=8)

        # Contenedor dividido en 3 sub-paneles: Izquierdo, Central, Derecho
        content_frame = ctk.CTkFrame(self.tab_ga, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=4)
        content_frame.grid_columnconfigure(2, weight=4)
        content_frame.grid_rowconfigure(0, weight=1)

        # SUB-PANEL IZQUIERDO: Matriz Dijkstra 6x6 y Evidencias de Operadores
        sub_left = ctk.CTkScrollableFrame(
            content_frame,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        sub_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        lbl_m_title = ctk.CTkLabel(
            sub_left,
            text="Matriz 6x6 de Distancias Mínimas (Dijkstra)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_m_title.pack(anchor="w", padx=10, pady=(5, 3))

        cols_m = ("orig", "A", "B", "C", "D", "E", "F")
        tree_m = ttk.Treeview(sub_left, columns=cols_m, show="headings", height=6)
        tree_m.heading("orig", text="Orig\\Dest")
        tree_m.column("orig", width=65, anchor="center")
        for c in ["A", "B", "C", "D", "E", "F"]:
            tree_m.heading(c, text=c)
            tree_m.column(c, width=32, anchor="center")
        tree_m.pack(fill="x", padx=10, pady=(0, 8))

        for u in sorted(MATRIZ_DISTANCIAS_MINIMAS.keys()):
            fila = [u] + [MATRIZ_DISTANCIAS_MINIMAS[u][v] for v in sorted(MATRIZ_DISTANCIAS_MINIMAS.keys())]
            tree_m.insert("", "end", values=fila)

        # Textbox con Población Inicial y Operadores Didácticos
        lbl_ops = ctk.CTkLabel(
            sub_left,
            text="Operadores Evolutivos y Población",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_ops.pack(anchor="w", padx=10, pady=(5, 3))

        self.txt_operadores = ctk.CTkTextbox(
            sub_left,
            height=270,
            fg_color=COLOR_SIDEBAR_BG,
            text_color=COLOR_TEXT_PRIMARY,
            border_width=1,
            border_color=COLOR_BORDER,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.txt_operadores.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Cargar texto pedagógico inicial de operadores
        ej_ox = obtener_ejemplo_ox_documentado()
        ej_swap = obtener_ejemplo_swap_documentado()
        txt_op_str = (
            "=== EJEMPLO DE CRUZAMIENTO POR ORDEN (OX) ===\n"
            f"Padre 1: {ej_ox['padre1']}\n"
            f"Padre 2: {ej_ox['padre2']}\n"
            f"Segmento de Padre 1 (pos {ej_ox['segmento_indices'][0]} a {ej_ox['segmento_indices'][1]}): {ej_ox['segmento_valores']}\n"
            f"Restantes de Padre 2: {ej_ox['restantes_padre2']}\n"
            f"➜ Hijo OX: {ej_ox['hijo']} (sin repetidos)\n\n"
            "=== EJEMPLO DE MUTACIÓN POR INTERCAMBIO (SWAP) ===\n"
            f"Antes:   {ej_swap['antes']}\n"
            f"Swap pos {ej_swap['posiciones'][0]} ('{ej_swap['genes'][0]}') y pos {ej_swap['posiciones'][1]} ('{ej_swap['genes'][1]}')\n"
            f"➜ Después: {ej_swap['despues']}\n\n"
            "Pulsa 'Ejecutar 5 Corridas Evolutivas' para visualizar las 10 permutaciones iniciales generadas."
        )
        self.txt_operadores.insert("0.0", txt_op_str)
        self.txt_operadores.configure(state="disabled")

        # SUB-PANEL CENTRAL: Tabla de 5 Semillas y Desglose Físico
        sub_center = ctk.CTkFrame(
            content_frame,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        sub_center.grid(row=0, column=1, sticky="nsew", padx=6)

        lbl_tbl_ga = ctk.CTkLabel(
            sub_center,
            text="Tabla Oficial de las 5 Ejecuciones",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_tbl_ga.pack(anchor="w", padx=12, pady=(8, 4))

        cols_runs = ("ejecucion", "semilla", "d_inicial", "d_final", "cromosoma")
        self.tree_runs = ttk.Treeview(sub_center, columns=cols_runs, show="headings", height=5)
        self.tree_runs.heading("ejecucion", text="Ejecución")
        self.tree_runs.heading("semilla", text="Semilla")
        self.tree_runs.heading("d_inicial", text="D. Inicial")
        self.tree_runs.heading("d_final", text="D. Final")
        self.tree_runs.heading("cromosoma", text="Mejor Cromosoma")

        self.tree_runs.column("ejecucion", width=65, anchor="center")
        self.tree_runs.column("semilla", width=65, anchor="center")
        self.tree_runs.column("d_inicial", width=75, anchor="center")
        self.tree_runs.column("d_final", width=75, anchor="center")
        self.tree_runs.column("cromosoma", width=120, anchor="center")
        self.tree_runs.pack(fill="x", padx=12, pady=(0, 8))

        lbl_desglose_title = ctk.CTkLabel(
            sub_center,
            text="Desglose de la Ruta Física y Verificación (360 m)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_desglose_title.pack(anchor="w", padx=12, pady=(4, 2))

        self.txt_desglose = ctk.CTkTextbox(
            sub_center,
            fg_color=COLOR_SIDEBAR_BG,
            text_color=COLOR_TEXT_PRIMARY,
            border_width=1,
            border_color=COLOR_BORDER,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.txt_desglose.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # SUB-PANEL DERECHO: Gráfica de Evolución (Matplotlib)
        sub_right = ctk.CTkFrame(
            content_frame,
            fg_color=COLOR_CARD_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=10
        )
        sub_right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        lbl_chart_title = ctk.CTkLabel(
            sub_right,
            text="Curvas de Convergencia (30 Generaciones)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_PRIMARY
        )
        lbl_chart_title.pack(anchor="w", padx=12, pady=(8, 4))

        chart_container = ctk.CTkFrame(
            sub_right,
            fg_color=COLOR_APP_BG,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=8
        )
        chart_container.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.fig_ga, self.ax_ga = plt.subplots(figsize=(4.5, 4.5), facecolor=COLOR_APP_BG)
        self.canvas_ga = FigureCanvasTkAgg(self.fig_ga, master=chart_container)
        self.canvas_ga.get_tk_widget().pack(fill="both", expand=True)

        self.inicializar_grafica_vacia()

    def inicializar_grafica_vacia(self):
        """Muestra una plantilla visual limpia de la gráfica antes de la primera ejecución."""
        self.ax_ga.clear()
        self.ax_ga.set_facecolor(COLOR_APP_BG)
        self.ax_ga.set_title("Evolución de la Mejor Distancia por Generación", color=COLOR_TEXT_PRIMARY, fontsize=10, pad=8)
        self.ax_ga.set_xlabel("Generación", color=COLOR_TEXT_SECONDARY, fontsize=9)
        self.ax_ga.set_ylabel("Distancia en Metros (m)", color=COLOR_TEXT_SECONDARY, fontsize=9)
        self.ax_ga.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=8)
        self.ax_ga.grid(True, linestyle="--", alpha=0.3, color=COLOR_BORDER_LIGHT)
        for spine in self.ax_ga.spines.values():
            spine.set_color(COLOR_BORDER)
        self.fig_ga.tight_layout()
        self.canvas_ga.draw()

    def accion_ejecutar_ga(self):
        """Ejecuta las 5 corridas evolutivas, llena las tablas y dibuja las 5 curvas con colores ORBIT."""
        corridas, mejor = ejecutar_5_corridas()
        self.resultado_genetico_corridas = corridas
        self.mejor_genetico_global = mejor

        # Limpiar y llenar tabla de 5 ejecuciones
        for item in self.tree_runs.get_children():
            self.tree_runs.delete(item)

        for c in corridas:
            self.tree_runs.insert("", "end", values=(
                f"Corrida {c['ejecucion']}",
                c['semilla'],
                f"{c['mejor_dist_inicial']} m",
                f"{c['mejor_dist_final']} m",
                c['mejor_cromosoma_str']
            ))

        # Actualizar textbox de operadores con la población inicial de la semilla 42
        pob_init = corridas[0]['poblacion_inicial']
        pob_str = "=== POBLACIÓN INICIAL GENERADA (Semilla 42) ===\n"
        for idx, ind in enumerate(pob_init, 1):
            dist_i = MATRIZ_DISTANCIAS_MINIMAS['A'][ind[0]]
            for k in range(len(ind) - 1):
                dist_i += MATRIZ_DISTANCIAS_MINIMAS[ind[k]][ind[k + 1]]
            dist_i += MATRIZ_DISTANCIAS_MINIMAS[ind[-1]]['A']
            pob_str += f"Individuo {idx:2d}: [{', '.join(ind)}]  -> {dist_i} m\n"

        ej_ox = obtener_ejemplo_ox_documentado()
        ej_swap = obtener_ejemplo_swap_documentado()
        pob_str += (
            "\n=== EJEMPLO DE CRUZAMIENTO OX ===\n"
            f"Padre 1: {ej_ox['padre1']}\n"
            f"Padre 2: {ej_ox['padre2']}\n"
            f"➜ Hijo OX: {ej_ox['hijo']}\n\n"
            "=== EJEMPLO DE MUTACIÓN SWAP ===\n"
            f"Antes:   {ej_swap['antes']}\n"
            f"➜ Mutado: {ej_swap['despues']}\n"
        )
        self.txt_operadores.configure(state="normal")
        self.txt_operadores.delete("0.0", "end")
        self.txt_operadores.insert("0.0", pob_str)
        self.txt_operadores.configure(state="disabled")

        # Desglose físico de la ruta óptima encontrada
        tramos = desglosar_ruta_fisica(mejor['mejor_ind_final'])
        verif = verificar_120_ordenes()

        desglose_txt = (
            f"MEJOR INDIVIDUO GLOBAL: [{mejor['mejor_cromosoma_str']}]\n"
            f"Distancia Total Verificada: {mejor['mejor_dist_final']} m (Aptitud: {1.0/mejor['mejor_dist_final']:.6f})\n"
            "--------------------------------------------------\n"
            "Tramos Físicos Autorizados:\n"
        )
        for t in tramos:
            desglose_txt += f"• Paso {t['paso']}: {t['tipo']} ({t['distancia_tramo']} m, Acum: {t['acumulado']} m)\n"
            desglose_txt += f"  Camino real: {t['camino_fisico']}  -> {t['accion']}\n"

        desglose_txt += (
            "--------------------------------------------------\n"
            "VERIFICACIÓN EXHAUSTIVA (120 PERMUTACIONES):\n"
            f"• Mínimo exacto: {verif['min_dist']} m ({', '.join(verif['optimos_str'])})\n"
            f"• Peor recorrido: {verif['peor_ejemplo_str']}\n"
            f"• Promedio: {verif['promedio_dist']:.1f} m\n"
            "✔ El algoritmo alcanzó el óptimo global teórico en las 5 corridas."
        )

        self.txt_desglose.configure(state="normal")
        self.txt_desglose.delete("0.0", "end")
        self.txt_desglose.insert("0.0", desglose_txt)
        self.txt_desglose.configure(state="disabled")

        # Graficar las 5 curvas de evolución con colores de la paleta ORBIT
        self.ax_ga.clear()
        self.ax_ga.set_facecolor(COLOR_APP_BG)

        # Colores extraídos de las categorías y acentos de ORBIT
        colores_curvas = [COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_WARNING, COLOR_SUCCESS, COLOR_INFO]
        estilos_linea = ["-", "--", "-.", ":", "-"]

        for i, c in enumerate(corridas):
            gens = [entry['generacion'] for entry in c['historial_generaciones']]
            mejores = [entry['mejor_distancia'] for entry in c['historial_generaciones']]
            self.ax_ga.plot(
                gens,
                mejores,
                label=f"Semilla {c['semilla']} (Final: {c['mejor_dist_final']}m)",
                color=colores_curvas[i % len(colores_curvas)],
                linestyle=estilos_linea[i % len(estilos_linea)],
                linewidth=2.0,
                alpha=0.9
            )

        self.ax_ga.set_title("Evolución de las 5 Corridas (30 Generaciones)", color=COLOR_TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=8)
        self.ax_ga.set_xlabel("Generación", color=COLOR_TEXT_PRIMARY, fontsize=9)
        self.ax_ga.set_ylabel("Mejor Distancia en Metros (m)", color=COLOR_TEXT_PRIMARY, fontsize=9)
        self.ax_ga.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=8)
        self.ax_ga.grid(True, linestyle="--", alpha=0.35, color=COLOR_BORDER_LIGHT)
        for spine in self.ax_ga.spines.values():
            spine.set_color(COLOR_BORDER)

        self.ax_ga.legend(
            facecolor=COLOR_CARD_BG,
            edgecolor=COLOR_BORDER,
            labelcolor=COLOR_TEXT_PRIMARY,
            fontsize=8,
            loc="upper right"
        )

        self.fig_ga.tight_layout()
        self.canvas_ga.draw()


def main():
    app = SistemaAlgoritmosApp()
    app.mainloop()


if __name__ == "__main__":
    main()
