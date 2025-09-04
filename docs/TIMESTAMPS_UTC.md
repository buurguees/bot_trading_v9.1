# Timestamps UTC en Logs

## 📅 Descripción

Se ha implementado un sistema automático para añadir timestamps UTC legibles a todos los logs del sistema. Ahora, además del timestamp en milisegundos, todos los logs incluyen campos UTC en formato legible para facilitar el análisis y debugging.

## 🕐 Formatos de Timestamp

### Timestamp Original
- **Formato**: Unix timestamp en milisegundos
- **Ejemplo**: `1756994668970`
- **Uso**: Para cálculos y comparaciones programáticas

### Timestamp UTC Legible
- **Formato**: `YYYY-MM-DD HH:MM:SS UTC`
- **Ejemplo**: `2025-09-04 14:04:28 UTC`
- **Uso**: Para lectura humana y análisis manual

### Timestamp ISO 8601
- **Formato**: `YYYY-MM-DDTHH:MM:SS.fffZ`
- **Ejemplo**: `2025-09-04T14:04:28.970000Z`
- **Uso**: Para integración con sistemas externos

## 📊 Logs Afectados

### 1. Logs de Runs (`models/{symbol}/{symbol}_runs.jsonl`)

**Campos añadidos automáticamente:**
- `ts_start_utc`: Timestamp de inicio del run en UTC
- `ts_start_iso`: Timestamp de inicio en formato ISO 8601
- `ts_end_utc`: Timestamp de fin del run en UTC
- `ts_end_iso`: Timestamp de fin en formato ISO 8601

**Ejemplo:**
```json
{
  "symbol": "BTCUSDT",
  "ts_start": 1756991068967,
  "ts_end": 1756994668967,
  "ts_start_utc": "2025-09-04 13:04:28 UTC",
  "ts_start_iso": "2025-09-04T13:04:28.967000Z",
  "ts_end_utc": "2025-09-04 14:04:28 UTC",
  "ts_end_iso": "2025-09-04T14:04:28.967000Z",
  "trades_count": 5,
  "win_rate_trades": 60.0
}
```

### 2. Logs de Estrategias (`models/{symbol}/{symbol}_strategies_provisional.jsonl`)

**Campos añadidos automáticamente:**
- `ts_utc`: Timestamp del evento en UTC
- `ts_iso`: Timestamp del evento en formato ISO 8601

**Ejemplo:**
```json
{
  "kind": "OPEN",
  "side": 1,
  "price": 50000.0,
  "ts": 1756992868968,
  "ts_utc": "2025-09-04 13:34:28 UTC",
  "ts_iso": "2025-09-04T13:34:28.968000Z",
  "segment_id": 0
}
```

### 3. Métricas de Entrenamiento (`logs/ppo_v1/{symbol}_metrics.jsonl`)

**Campos añadidos automáticamente:**
- `ts_utc`: Timestamp de la métrica en UTC
- `ts_iso`: Timestamp de la métrica en formato ISO 8601

**Ejemplo:**
```json
{
  "ts": 1756994668970,
  "symbol": "BTCUSDT",
  "mode": "train",
  "ts_utc": "2025-09-04 14:04:28 UTC",
  "ts_iso": "2025-09-04T14:04:28.970000Z",
  "fps": 125.5,
  "total_timesteps": 204800
}
```

### 4. Métricas de Trades

**Campos añadidos automáticamente:**
- `first_trade_ts_utc`: Timestamp del primer trade en UTC
- `first_trade_ts_iso`: Timestamp del primer trade en formato ISO 8601
- `last_trade_ts_utc`: Timestamp del último trade en UTC
- `last_trade_ts_iso`: Timestamp del último trade en formato ISO 8601

**Ejemplo:**
```json
{
  "trades_count": 10,
  "first_trade_ts": 1756987468971,
  "last_trade_ts": 1756994668971,
  "first_trade_ts_utc": "2025-09-04 12:04:28 UTC",
  "first_trade_ts_iso": "2025-09-04T12:04:28.971000Z",
  "last_trade_ts_utc": "2025-09-04 14:04:28 UTC",
  "last_trade_ts_iso": "2025-09-04T14:04:28.971000Z"
}
```

## 🔧 Implementación Técnica

### Archivos Modificados

1. **`base_env/utils/timestamp_utils.py`** (NUEVO)
   - Funciones utilitarias para conversión de timestamps
   - `timestamp_to_utc_string()`: Convierte a formato legible
   - `timestamp_to_utc_iso()`: Convierte a formato ISO 8601
   - `add_utc_timestamps()`: Añade campos UTC automáticamente

2. **`base_env/logging/run_logger.py`**
   - Añade timestamps UTC a logs de runs
   - Usa `add_utc_timestamps()` antes de guardar

3. **`train_env/strategy_logger.py`**
   - Añade timestamps UTC a logs de estrategias
   - Usa `add_utc_timestamps()` antes de guardar

4. **`train_env/callbacks/training_metrics_callback.py`**
   - Añade timestamps UTC a métricas de entrenamiento
   - Usa `add_utc_timestamps()` antes de guardar

5. **`base_env/metrics/trade_metrics.py`**
   - Añade propiedades UTC a `TradeRecord`
   - Incluye timestamps UTC en métricas calculadas

### Campos de Timestamp Reconocidos

La función `add_utc_timestamps()` reconoce automáticamente estos campos:
- `ts`, `timestamp`
- `ts_start`, `ts_end`
- `open_ts`, `close_ts`
- `first_trade_ts`, `last_trade_ts`
- `created_at`, `updated_at`

## 🎯 Beneficios

### Para Análisis Manual
- **Legibilidad**: Fechas y horas en formato humano
- **Debugging**: Fácil identificación de eventos en el tiempo
- **Análisis**: Correlación con eventos del mercado

### Para Integración
- **ISO 8601**: Formato estándar para sistemas externos
- **UTC**: Zona horaria consistente
- **Precisión**: Milisegundos para análisis detallado

### Para Desarrollo
- **Automatización**: Se añaden automáticamente
- **Consistencia**: Mismo formato en todos los logs
- **Retrocompatibilidad**: Los timestamps originales se mantienen

## 📝 Ejemplos de Uso

### Verificar Timestamps en Logs
```bash
# Ver logs de runs con timestamps UTC
cat models/BTCUSDT/BTCUSDT_runs.jsonl | jq '.ts_start_utc, .ts_end_utc'

# Ver logs de estrategias con timestamps UTC
cat models/BTCUSDT/BTCUSDT_strategies_provisional.jsonl | jq '.ts_utc'
```

### Análisis Temporal
```python
from base_env.utils.timestamp_utils import timestamp_to_utc_string

# Convertir timestamp manualmente
timestamp = 1756994668970
utc_string = timestamp_to_utc_string(timestamp)
print(f"Timestamp {timestamp} = {utc_string}")
# Output: Timestamp 1756994668970 = 2025-09-04 14:04:28 UTC
```

### Filtrado por Fecha
```bash
# Buscar trades de un día específico
cat models/BTCUSDT/BTCUSDT_strategies_provisional.jsonl | \
jq 'select(.ts_utc | startswith("2025-09-04"))'
```

## ⚠️ Notas Importantes

1. **Zona Horaria**: Todos los timestamps UTC están en UTC (GMT+0)
2. **Precisión**: Se mantiene la precisión de milisegundos
3. **Rendimiento**: La conversión es mínima y no afecta el rendimiento
4. **Compatibilidad**: Los timestamps originales se mantienen intactos
5. **Automatización**: No requiere cambios en el código existente

## 🔍 Troubleshooting

### Timestamps Inválidos
Si un timestamp es `None` o inválido, los campos UTC correspondientes serán `None`.

### Formato de Fecha
Los timestamps UTC siguen el formato estándar: `YYYY-MM-DD HH:MM:SS UTC`

### Zona Horaria
Todos los timestamps están en UTC. Para convertir a zona horaria local, usar herramientas externas.
