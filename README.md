# 🤖 Bot Trading v9.1 - Sistema de Trading RL Optimizado

## 🚀 **Descripción General**

Sistema de trading automatizado basado en Reinforcement Learning (RL) con PPO, optimizado para 50M steps de entrenamiento. Incluye modo autonomía total, configuración centralizada desde YAML, y sistema de rewards/penalties avanzado.

## ✨ **Características Principales**

### **🎯 Modo Autonomía Total**
- ✅ **Configuración 100% desde YAML** - Sin parámetros hardcodeados
- ✅ **Carga automática** de configuración desde `config/`
- ✅ **Validación de duplicados** y consistencia
- ✅ **BaseTradingEnv autónomo** con `from_yaml_dir()`

### **⚡ Optimizaciones para 50M Steps**
- ✅ **Sistema de rewards optimizado** con clipping dinámico
- ✅ **Procesamiento vectorizado** para entornos paralelos
- ✅ **Cache inteligente** con TTL y validación
- ✅ **Profiling de rendimiento** en tiempo real
- ✅ **Curriculum learning** para diferentes etapas

### **🔧 Arquitectura Modular**
- ✅ **Utilidades centralizadas** sin duplicación de código
- ✅ **Módulos de rewards** especializados y optimizados
- ✅ **Sistema de validación** robusto
- ✅ **Herramientas de monitoreo** integradas

## 📁 **Estructura del Proyecto**

```
bot_trading_v9.1/
├── app.py                          # ← Punto de entrada principal
├── requirements.txt                 # ← Dependencias del proyecto
├── COMANDOS_SISTEMA.txt            # ← Comandos y guías de uso
├── README.md                       # ← Este archivo
├── __init__.py                     # ← Inicialización del paquete
│
├── base_env/                       # ← Entorno base de trading
│   ├── base_env.py                 # ← Entorno principal optimizado
│   ├── config/                     # ← Configuración centralizada
│   │   ├── config_loader.py        # ← Cargador de YAMLs
│   │   └── config_utils.py         # ← Utilidades centralizadas
│   ├── actions/                    # ← Sistema de rewards/penalties
│   │   ├── reward_orchestrator_optimized.py
│   │   ├── time_efficiency_reward.py
│   │   ├── reward_decomposition.py
│   │   └── rewards_utils.py        # ← Utilidades de rewards
│   ├── io/                         # ← Brokers y conectores
│   ├── features/                   # ← Pipeline de indicadores
│   ├── analysis/                   # ← Análisis jerárquico
│   ├── policy/                     # ← Motor de decisiones
│   ├── risk/                       # ← Gestión de riesgo
│   └── accounting/                 # ← Contabilidad y PnL
│
├── train_env/                      # ← Entorno de entrenamiento
│   ├── core/                       # ← Núcleo de entrenamiento
│   ├── callbacks/                  # ← Callbacks de PPO
│   ├── analysis/                   # ← Análisis de rendimiento
│   └── optimization/               # ← Optimizaciones
│
├── config/                         # ← Configuración YAML
│   ├── settings.yaml              # ← Configuración global
│   ├── symbols.yaml               # ← Símbolos y mercados
│   ├── train.yaml                 # ← Configuración de entrenamiento
│   ├── rewards_optimized.yaml     # ← Sistema de rewards
│   └── risk.yaml                  # ← Gestión de riesgo
│
├── scripts/                        # ← Scripts de utilidad
│   ├── train_ppo.py               # ← Entrenamiento principal
│   ├── validate_rewards_system.py # ← Validación del sistema
│   ├── benchmark_rewards.py       # ← Benchmark de rendimiento
│   └── clean_yaml_duplicates.py   # ← Limpieza de duplicados
│
├── tests/                          # ← Suite de pruebas
│   ├── unit/                       # ← Pruebas unitarias
│   ├── integration/                # ← Pruebas de integración
│   └── e2e/                        # ← Pruebas end-to-end
│
├── data/                           # ← Datos históricos
├── models/                         # ← Modelos entrenados
├── logs/                           # ← Logs del sistema
├── monitoring/                     # ← Monitoreo en tiempo real
│
├── archive/                        # ← Archivos históricos
├── docs_backup/                    # ← Documentación de respaldo
├── scripts_backup/                 # ← Scripts de respaldo
└── utils_backup/                   # ← Utilidades de respaldo
```

## 🚀 **Inicio Rápido**

### **1. Instalación**
```bash
# Clonar el repositorio
git clone <repository-url>
cd bot_trading_v9.1

# Instalar dependencias
pip install -r requirements.txt
```

### **2. Configuración**
```bash
# Validar configuración YAML
python scripts/clean_yaml_duplicates.py

# Verificar sistema de rewards
python scripts/validate_rewards_system.py
```

### **3. Entrenamiento**
```bash
# Entrenamiento con modo autonomía
python scripts/train_ppo.py

# O usar la app principal
python app.py
```

## 🔧 **Uso del Modo Autonomía**

### **Crear Entorno Autónomo**
```python
from base_env.base_env import BaseTradingEnv
from base_env.io.historical_broker import ParquetHistoricalBroker

# Crear entorno completamente autónomo
env = BaseTradingEnv.from_yaml_dir(
    config_dir="config/",
    broker=broker,
    oms=oms,
    models_root="models",
    antifreeze_enabled=False
)
```

### **Configuración desde YAML**
El sistema lee automáticamente:
- **Símbolos y mercados** desde `symbols.yaml`
- **Timeframes y datos** desde `train.yaml`
- **Sistema de rewards** desde `rewards_optimized.yaml`
- **Gestión de riesgo** desde `risk.yaml`
- **Configuración global** desde `settings.yaml`

## ⚡ **Optimizaciones Implementadas**

### **Sistema de Rewards Optimizado**
- **Clipping dinámico** basado en estadísticas históricas
- **Normalización global** para estabilidad en PPO
- **Procesamiento vectorizado** para entornos paralelos
- **Cache inteligente** con TTL de 5 minutos
- **Profiling de rendimiento** en tiempo real

### **Mejoras de Rendimiento**
- **3-10x más rápido** que la versión original
- **60% menos código** con utilidades centralizadas
- **Escalabilidad lineal** con número de entornos
- **Memoria controlada** para 50M steps

### **Herramientas de Monitoreo**
- **Validación automática** de configuración
- **Detección de duplicados** en YAML
- **Benchmark de rendimiento** integrado
- **Reportes de validación** detallados

## 📊 **Métricas de Rendimiento**

### **Tiempo Estimado para 50M Steps:**
- **Original**: ~8-16 horas
- **Optimizado**: ~2-5 horas
- **Batch (4 envs)**: ~1-3 horas
- **Tiempo Ahorrado**: 6-13 horas

### **Uso de Memoria:**
- **Controlado**: <1GB para 50M steps
- **Limpieza automática** de datos antiguos
- **Cache eficiente** con TTL

## 🛠️ **Comandos Útiles**

### **Validación del Sistema**
```bash
# Validar configuración YAML
python scripts/clean_yaml_duplicates.py

# Validar sistema de rewards
python scripts/validate_rewards_system.py

# Benchmark de rendimiento
python scripts/benchmark_rewards.py
```

### **Entrenamiento**
```bash
# Entrenamiento estándar
python scripts/train_ppo.py

# Entrenamiento con monitoreo
python monitoring/monitor_training.py
```

### **Ejemplos**
```bash
# Modo autonomía
python scripts/example_autonomous_mode.py
```

## 📋 **Requisitos del Sistema**

### **Dependencias Principales**
- Python 3.8+
- PyTorch 1.12+
- Stable-Baselines3 2.0+
- NumPy 1.21+
- Pandas 1.5+
- PyYAML 6.0+

### **Requisitos de Hardware**
- **RAM**: 8GB mínimo, 16GB recomendado
- **GPU**: Opcional, pero recomendada para entrenamiento
- **Almacenamiento**: 10GB para datos históricos
- **CPU**: 4+ cores recomendados

## 🔍 **Troubleshooting**

### **Problemas Comunes**
1. **Error de configuración**: Ejecutar `python scripts/clean_yaml_duplicates.py`
2. **Error de rewards**: Ejecutar `python scripts/validate_rewards_system.py`
3. **Error de memoria**: Reducir `n_envs` en configuración
4. **Error de datos**: Verificar que `data/` contenga datos históricos

### **Logs y Debugging**
- **Logs de entrenamiento**: `logs/ppo_v1/`
- **Logs de validación**: `validation_results.json`
- **Logs de rendimiento**: `performance_results.json`

## 📚 **Documentación Adicional**

- **Modo Autonomía**: `docs_backup/MODO_AUTONOMIA_TOTAL.md`
- **Consolidación**: `docs_backup/CONSOLIDACION_FUNCIONES_DUPLICADAS.md`
- **Correcciones**: `docs_backup/RESUMEN_FINAL_CORRECCIONES.md`
- **Comandos**: `COMANDOS_SISTEMA.txt`

## 🤝 **Contribución**

1. Fork el repositorio
2. Crear rama para feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 **Licencia**

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 **Soporte**

Para soporte técnico o preguntas:
- Crear un issue en GitHub
- Revisar la documentación en `docs_backup/`
- Ejecutar scripts de validación para diagnóstico

---

**Desarrollado con ❤️ para trading algorítmico con RL**