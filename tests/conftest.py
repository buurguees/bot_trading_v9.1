from __future__ import annotations

import os
import sys

# Asegura que el directorio del proyecto esté en sys.path para los imports de tests
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


