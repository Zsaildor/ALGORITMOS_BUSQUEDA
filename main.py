"""
Punto de Entrada Principal del Mini Sistema de Búsqueda y Optimización (IA)
Ejecuta la interfaz gráfica construida en CustomTkinter.
"""

import sys
import os

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.main_gui import main

if __name__ == "__main__":
    main()
