# 📁 Estructura de train_env

## 🏗️ **Organización Modular del Sistema de Entrenamiento**

La carpeta `train_env` ha sido reorganizada de forma lógica para mejorar la mantenibilidad y claridad del código.

### **📂 Estructura de Directorios**

```
train_env/
├── core/                    # 🎯 Componentes principales
│   ├── training_orchestrator.py    # Orquestador principal
│   ├── gym_wrapper.py              # Wrapper para Gymnasium
│   ├── vec_factory.py              # Factory unificada (cronológico + simple)
│   ├── model_manager.py            # Gestor de modelos
│   └── worker_manager.py           # Gestor inteligente de workers
│
├── scripts/                 # 📜 Scripts de entrenamiento
│   ├── train_ppo.py                # Script principal de entrenamiento
│   ├── monitor_training.py         # Monitoreo de entrenamiento
│   ├── monitor_performance.py      # Monitoreo de performance
│   └── smoke_run.py               # Pruebas rápidas
│
├── optimization/            # ⚡ Optimización y tuning
│   ├── tune_parameters.py          # Tuning de hiperparámetros
│   ├── quick_tuning.py             # Tuning rápido
│   └── phase_activation.py         # Activación de fases
│
├── analysis/                # 📊 Análisis y métricas
│   ├── analyze_results.py          # Análisis de resultados
│   ├── check_best_run.py           # Verificación de mejores runs
│   ├── show_progress.py            # Visualización de progreso
│   └── watch_progress.py           # Monitoreo en tiempo real
│
├── utilities/               # 🔧 Utilidades y herramientas
│   ├── dataset.py                  # Creación de datasets
│   ├── reward_shaper.py            # Moldeado de recompensas
│   ├── strategy_aggregator.py      # Agregación de estrategias
│   ├── strategy_curriculum.py      # Currículo de estrategias
│   ├── strategy_logger.py          # Logger de estrategias
│   ├── strategy_persistence.py     # Persistencia de estrategias
│   ├── learning_rate_reset_callback.py  # Reset de learning rate
│   ├── repair_models.py            # Reparación de modelos
│   ├── clean_duplicate_runs.py     # Limpieza de runs duplicados
│   ├── sanity_checks.py            # Verificaciones de sanidad
│   └── validate_compatibility.py   # Validación de compatibilidad
│
├── config/                  # ⚙️ Configuraciones específicas
│   └── (archivos de configuración específicos)
│
├── callbacks/               # 🔄 Callbacks de entrenamiento
│   ├── periodic_checkpoint.py      # Checkpoints periódicos
│   ├── strategy_keeper.py          # Guardado de estrategias
│   ├── strategy_consultant.py      # Consultor de estrategias
│   ├── anti_bad_strategy.py        # Anti-estrategias malas
│   ├── main_model_saver.py         # Guardado de modelo principal
│   └── training_metrics_callback.py # Métricas de entrenamiento
│
├── monitoring/              # 📈 Monitoreo en tiempo real
│   ├── real_time_monitor.py        # Monitor principal
│   └── (otros componentes de monitoreo)
│
└── utils/                   # 🛠️ Utilidades generales
    └── sanitizers.py               # Sanitizadores de datos
```

### **🎯 Propósito de Cada Directorio**

#### **core/** - Componentes Principales
- **training_orchestrator.py**: Coordina todo el proceso de entrenamiento
- **gym_wrapper.py**: Adapta el entorno a la API de Gymnasium
- **vec_factory_*.py**: Crea entornos vectorizados para paralelización
- **model_manager.py**: Gestiona modelos, checkpoints y estrategias
- **worker_manager.py**: Optimiza el número de workers basado en recursos

#### **scripts/** - Scripts de Entrenamiento
- **train_ppo.py**: Script principal de entrenamiento (refactorizado)
- **monitor_*.py**: Scripts de monitoreo y observación
- **smoke_run.py**: Pruebas rápidas y validación

#### **optimization/** - Optimización
- **tune_parameters.py**: Tuning automático de hiperparámetros
- **quick_tuning.py**: Tuning rápido para pruebas
- **phase_activation.py**: Gestión de fases de entrenamiento

#### **analysis/** - Análisis
- **analyze_results.py**: Análisis detallado de resultados
- **check_best_run.py**: Verificación de mejores runs
- **show_progress.py**: Visualización de progreso
- **watch_progress.py**: Monitoreo en tiempo real

#### **utilities/** - Utilidades
- **dataset.py**: Creación y gestión de datasets
- **reward_shaper.py**: Moldeado de recompensas
- **strategy_*.py**: Gestión de estrategias
- **repair_models.py**: Reparación de archivos corruptos
- **sanity_checks.py**: Verificaciones de integridad

#### **callbacks/** - Callbacks
- Callbacks específicos para diferentes aspectos del entrenamiento
- Manejo de checkpoints, estrategias y métricas

#### **monitoring/** - Monitoreo
- Sistema de monitoreo en tiempo real
- Alertas y métricas de performance

### **🔄 Compatibilidad**

Para mantener la compatibilidad, se ha creado un script de conveniencia en `scripts/train_ppo.py` que redirige a la nueva ubicación.

### **📦 Imports**

Los imports han sido actualizados para usar la nueva estructura:

```python
# Antes
from train_env.training_orchestrator import TrainingOrchestrator
from train_env.model_manager import ModelManager

# Ahora
from train_env.core.training_orchestrator import TrainingOrchestrator
from train_env.core.model_manager import ModelManager
```

O usando el import simplificado:

```python
from train_env import TrainingOrchestrator, ModelManager
```

### **✅ Beneficios de la Reorganización**

1. **Claridad**: Cada directorio tiene un propósito específico
2. **Mantenibilidad**: Fácil localizar y modificar componentes
3. **Escalabilidad**: Estructura preparada para crecimiento
4. **Modularidad**: Componentes independientes y reutilizables
5. **Organización**: Separación clara de responsabilidades
