# 📁 ESTRUCTURA DEL PROYECTO - Bot Trading v9.1

## 🎯 **DIRECTORIO PRINCIPAL (LIMPIO)**

```
bot_trading_v9.1/
├── 📄 app.py                          # ← Punto de entrada principal
├── 📄 requirements.txt                 # ← Dependencias del proyecto
├── 📄 COMANDOS_SISTEMA.txt            # ← Comandos y guías de uso
├── 📄 README.md                       # ← Documentación principal
├── 📄 __init__.py                     # ← Inicialización del paquete
├── 📄 .gitignore                      # ← Archivos ignorados por Git
└── 📄 ESTRUCTURA_PROYECTO.md          # ← Este archivo
```

## 🏗️ **MÓDULOS PRINCIPALES**

### **`base_env/` - Entorno Base de Trading**
```
base_env/
├── base_env.py                        # ← Entorno principal optimizado
├── config/                            # ← Configuración centralizada
│   ├── config_loader.py              # ← Cargador de YAMLs
│   ├── config_utils.py               # ← Utilidades centralizadas
│   ├── config_validator.py           # ← Validador de configuración
│   ├── models.py                     # ← Modelos de datos
│   └── symbols_loader.py             # ← Cargador de símbolos
├── actions/                           # ← Sistema de rewards/penalties
│   ├── reward_orchestrator_optimized.py
│   ├── time_efficiency_reward.py
│   ├── reward_decomposition.py
│   ├── exploration_bonus.py
│   ├── rewards_utils.py              # ← Utilidades de rewards
│   └── [otros módulos de rewards]
├── io/                                # ← Brokers y conectores
│   ├── broker.py
│   ├── historical_broker.py
│   ├── live_ws.py
│   └── [otros conectores]
├── features/                          # ← Pipeline de indicadores
│   ├── pipeline.py
│   ├── indicators.py
│   └── README.md
├── analysis/                          # ← Análisis jerárquico
│   ├── hierarchical.py
│   └── README.md
├── policy/                            # ← Motor de decisiones
│   ├── gating.py
│   ├── rules.py
│   └── README.md
├── risk/                              # ← Gestión de riesgo
│   ├── manager.py
│   └── README.md
├── accounting/                        # ← Contabilidad y PnL
│   ├── ledger.py
│   ├── fees.py
│   └── README.md
├── events/                            # ← Sistema de eventos
│   ├── domain.py
│   ├── bus.py
│   └── README.md
├── tfs/                               # ← Alineación multi-timeframe
│   ├── alignment.py
│   └── README.md
├── smc/                               # ← Detección SMC
│   ├── detector.py
│   └── README.md
├── logging/                           # ← Sistema de logging
├── telemetry/                         # ← Telemetría
├── metrics/                           # ← Métricas
└── utils/                             # ← Utilidades generales
```

### **`train_env/` - Entorno de Entrenamiento**
```
train_env/
├── core/                              # ← Núcleo de entrenamiento
│   ├── training_orchestrator.py
│   ├── worker_manager.py
│   └── [otros módulos core]
├── callbacks/                         # ← Callbacks de PPO
│   ├── main_model_saver.py
│   ├── strategy_consultant.py
│   └── [otros callbacks]
├── analysis/                          # ← Análisis de rendimiento
│   ├── show_progress.py
│   └── [otros análisis]
├── optimization/                      # ← Optimizaciones
│   ├── hyperparameter_tuner.py
│   └── [otras optimizaciones]
├── monitoring/                        # ← Monitoreo
├── utilities/                         # ← Utilidades de entrenamiento
└── scripts/                           # ← Scripts de entrenamiento
```

### **`config/` - Configuración YAML**
```
config/
├── settings.yaml                      # ← Configuración global
├── symbols.yaml                       # ← Símbolos y mercados
├── train.yaml                         # ← Configuración de entrenamiento
├── rewards_optimized.yaml             # ← Sistema de rewards optimizado
├── rewards.yaml                       # ← Sistema de rewards original
├── risk.yaml                          # ← Gestión de riesgo
├── risk_commented.yaml                # ← Documentación de riesgo
├── pipeline.yaml                      # ← Pipeline de indicadores
├── hierarchical.yaml                  # ← Análisis jerárquico
├── oms.yaml                           # ← Configuración OMS
├── fees.yaml                          # ← Configuración de fees
└── README.md                          # ← Documentación de configuración
```

### **`scripts/` - Scripts de Utilidad**
```
scripts/
├── train_ppo.py                       # ← Entrenamiento principal
├── validate_rewards_system.py         # ← Validación del sistema
├── benchmark_rewards.py               # ← Benchmark de rendimiento
├── clean_yaml_duplicates.py           # ← Limpieza de duplicados
├── example_autonomous_mode.py         # ← Ejemplo modo autonomía
└── [otros scripts de utilidad]
```

### **`tests/` - Suite de Pruebas**
```
tests/
├── unit/                              # ← Pruebas unitarias
│   ├── test_rewards.py
│   ├── test_config.py
│   └── [otras pruebas unitarias]
├── integration/                       # ← Pruebas de integración
│   ├── test_training_integration.py
│   ├── test_rewards_integration.py
│   └── [otras pruebas de integración]
├── e2e/                               # ← Pruebas end-to-end
│   ├── test_full_training.py
│   └── [otras pruebas e2e]
├── conftest.py                        # ← Configuración de pytest
└── README.md                          # ← Documentación de pruebas
```

## 📊 **DATOS Y MODELOS**

### **`data/` - Datos Históricos**
```
data/
├── BTCUSDT/                           # ← Datos de BTCUSDT
│   ├── 1m/                           # ← Datos de 1 minuto
│   ├── 5m/                           # ← Datos de 5 minutos
│   └── [otros timeframes]
├── _TEMPLATE_SYMBOL_/                 # ← Template para nuevos símbolos
└── README.md                          # ← Documentación de datos
```

### **`models/` - Modelos Entrenados**
```
models/
├── BTCUSDT/                           # ← Modelos de BTCUSDT
│   ├── ppo_v1/                       # ← Modelos PPO v1
│   ├── final_model.zip               # ← Modelo final
│   └── [otros modelos]
├── TEST/                              # ← Modelos de prueba
└── tmp/                               # ← Modelos temporales
```

### **`logs/` - Logs del Sistema**
```
logs/
├── ppo_v1/                            # ← Logs de entrenamiento PPO
├── test_logs/                         # ← Logs de pruebas
└── [otros logs]
```

## 🔧 **HERRAMIENTAS Y MONITOREO**

### **`monitoring/` - Monitoreo en Tiempo Real**
```
monitoring/
├── monitor_training.py                # ← Monitoreo de entrenamiento
├── monitor_actions.py                 # ← Monitoreo de acciones
├── monitor_logs.py                    # ← Monitoreo de logs
├── monitor_fixes.py                   # ← Monitoreo de correcciones
├── start_training.ps1                 # ← Script PowerShell
└── monitor_training.ps1               # ← Script PowerShell
```

### **`data_pipeline/` - Pipeline de Datos**
```
data_pipeline/
├── collectors/                        # ← Colectores de datos
├── scripts/                           # ← Scripts de procesamiento
├── schemas/                           # ← Esquemas de datos
└── docs/                              # ← Documentación
```

## 📚 **DOCUMENTACIÓN Y ARCHIVOS**

### **`docs_backup/` - Documentación de Respaldo**
```
docs_backup/
├── MODO_AUTONOMIA_TOTAL.md            # ← Documentación modo autonomía
├── CONSOLIDACION_FUNCIONES_DUPLICADAS.md
├── RESUMEN_FINAL_CORRECCIONES.md
├── [otra documentación]
└── README.md
```

### **`archive/` - Archivos Históricos**
```
archive/
├── bot_trading_v9.1.0.zip            # ← Versiones anteriores
├── bot_trading_v9.1.1.zip
├── [otras versiones]
├── validation_results.json            # ← Resultados de validación
├── performance_results.json           # ← Resultados de rendimiento
└── compatibility_results.json         # ← Resultados de compatibilidad
```

### **`scripts_backup/` - Scripts de Respaldo**
```
scripts_backup/
├── [scripts movidos del directorio raíz]
└── [scripts de respaldo]
```

### **`utils_backup/` - Utilidades de Respaldo**
```
utils_backup/
├── [utilidades movidas del directorio raíz]
└── [utilidades de respaldo]
```

## 🎯 **PRINCIPIOS DE ORGANIZACIÓN**

### **✅ Archivos en Directorio Raíz (Solo Esenciales)**
- `app.py` - Punto de entrada principal
- `requirements.txt` - Dependencias
- `COMANDOS_SISTEMA.txt` - Comandos y guías
- `README.md` - Documentación principal
- `__init__.py` - Inicialización del paquete
- `.gitignore` - Archivos ignorados por Git

### **✅ Carpetas Organizadas por Función**
- `base_env/` - Entorno base de trading
- `train_env/` - Entorno de entrenamiento
- `config/` - Configuración YAML
- `scripts/` - Scripts de utilidad
- `tests/` - Suite de pruebas
- `data/` - Datos históricos
- `models/` - Modelos entrenados
- `logs/` - Logs del sistema
- `monitoring/` - Monitoreo en tiempo real

### **✅ Archivos de Respaldo Organizados**
- `archive/` - Archivos históricos y versiones
- `docs_backup/` - Documentación de respaldo
- `scripts_backup/` - Scripts de respaldo
- `utils_backup/` - Utilidades de respaldo

## 🚀 **BENEFICIOS DE LA ORGANIZACIÓN**

1. **Directorio raíz limpio** - Solo archivos esenciales
2. **Estructura modular** - Fácil navegación y mantenimiento
3. **Separación clara** - Código, datos, configuración y documentación
4. **Archivos de respaldo** - Organizados y accesibles
5. **Documentación centralizada** - Fácil de encontrar y mantener
6. **Escalabilidad** - Fácil añadir nuevos módulos

---

**Estructura optimizada para desarrollo, mantenimiento y escalabilidad** 🎯
