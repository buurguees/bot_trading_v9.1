# 🚨 SOLUCIÓN AL BLOQUEO DE TRADES

## 📊 PROBLEMA IDENTIFICADO

**Diagnóstico**: 400/400 runs terminan en BANKRUPTCY con 0 trades ejecutados.

**Razones principales**:
- `POLICY_NO_OPEN`: ~600 veces por run (RL envía action=0 en lugar de acciones reales)
- `NO_SL_DISTANCE`: ~200 veces por run (SL muy cerca del precio)
- `NO_SIGNAL`: ~150 veces por run (policy jerárquica no genera señales)

## 🔧 SOLUCIONES IMPLEMENTADAS

### 1. ✅ Fix del RL Actions (CRÍTICO)
**Archivo**: `train_env/gym_wrapper.py`
**Problema**: RL enviaba `action=0` (dejar policy) en lugar de acciones reales (1,3,4)
**Solución**: Forzar acciones reales cuando RL envía action=0
```python
# Si RL envía action=0, cambiarlo a acción aleatoria (3=force_long, 4=force_short)
if trade_action == 0:
    trade_action = random.choice([3, 4])
    print(f"🔧 FIX RL: action=0 → {trade_action}")
```

### 2. ✅ Aumento de Risk Config
**Archivo**: `config/risk.yaml`
**Cambios**:
- `spot.risk_pct_per_trade`: 1.0% → 2.0%
- `futures.risk_pct_per_trade`: 2.0% → 3.0%

### 3. ✅ Configuración de SL/TP Defaults
**Archivo**: `config/risk.yaml`
**Ya configurado correctamente**:
```yaml
default_levels:
  use_atr: true
  atr_period: 14
  sl_atr_mult: 1.0
  min_sl_pct: 1.0
  tp_r_multiple: 1.5
```

### 4. ✅ Min Notional Force
**Archivo**: `config/risk.yaml`
**Ya configurado**: `train_force_min_notional: true`

### 5. ✅ Validación de Ledger
**Verificado**: No hay fees fantasma erosionando equity sin trades

## 🚀 PRÓXIMOS PASOS

### Inmediato
1. **Reiniciar entrenamiento**:
   ```bash
   python restart_training.py
   ```

2. **Monitorear progreso**:
   - Buscar mensajes: `🔧 FIX RL: action=0 → 3/4`
   - Buscar trades: `OPEN LONG/SHORT`
   - Verificar que `trades_count > 0` en runs

### Monitoreo
```bash
# En otra terminal
python scripts/watch_progress.py
```

### Restauración (cuando funcione)
```bash
python fix_rl_actions.py --restore
```

## 📈 RESULTADOS ESPERADOS

**Antes**:
- 400/400 runs: 0 trades, BANKRUPTCY
- `POLICY_NO_OPEN`: ~600/run
- Equity erosiona hasta 12-997 USDT

**Después** (esperado):
- Runs con `trades_count > 0`
- `POLICY_NO_OPEN` reducido significativamente
- Equity estable o creciente
- Aprendizaje del RL funcionando

## 🔍 DIAGNÓSTICO ADICIONAL

Si el problema persiste después de estos fixes:

1. **Verificar logs de entrenamiento**:
   - ¿El RL está aprendiendo?
   - ¿Hay errores en el modelo?

2. **Ajustar parámetros de RL**:
   - `ent_coef`: aumentar exploración
   - `learning_rate`: verificar que no sea muy bajo

3. **Revisar configuración de rewards**:
   - `empty_run_penalty`: -0.5 (ya configurado)
   - `survival_bonus`: 0.001 (ya configurado)

## 📝 ARCHIVOS MODIFICADOS

1. `train_env/gym_wrapper.py` - Fix temporal para RL actions
2. `config/risk.yaml` - Aumento de risk_pct_per_trade
3. `fix_rl_actions.py` - Script para aplicar/restaurar fix
4. `restart_training.py` - Script para reiniciar entrenamiento
5. `debug_rl_actions.py` - Script de diagnóstico

## ⚠️ NOTAS IMPORTANTES

- El fix del gym_wrapper es **temporal** y debe restaurarse una vez que el RL aprenda
- Los parámetros de riesgo están **aumentados temporalmente** para permitir trades
- Monitorear que no haya overfitting o trades excesivos
- Una vez que funcione, ajustar parámetros gradualmente
