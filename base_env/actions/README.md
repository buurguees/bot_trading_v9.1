# Sistema Modular de Rewards y Penalties

Este directorio contiene un sistema modular de rewards y penalties para el trading algorítmico. Cada componente está en su propio archivo para facilitar el mantenimiento y la configuración.

## Estructura

### Sistemas Individuales

1. **`take_profit_reward.py`** - Rewards por Take Profit hits
   - Reward base por TP: +1.0
   - Efficient R/R bonus: +0.2 si TP con drawdown < 50% del SL
   - Bonus por R-multiple alto y ROI positivo

2. **`stop_loss_penalty.py`** - Penalties por Stop Loss hits
   - Penalty base por SL: -0.5
   - Bonus/malus configurable por SL_HIT

3. **`bankruptcy_penalty.py`** - Penalties por Bankruptcy
   - Penalty severo por bancarrota: -10.0
   - Bonus por supervivencia: +0.001 cada step

4. **`holding_reward.py`** - Rewards por Holding
   - Reward por mantener posiciones con equity positivo: +0.1 cada 10 barras
   - Contador de barras con equity positivo

5. **`inactivity_penalty.py`** - Penalties por Inactividad
   - Penalty por no abrir trades: -0.01 cada 100 pasos
   - Contador de pasos desde último trade

6. **`roi_reward.py`** - Rewards por ROI
   - Sistema de tramos configurables para ROI positivo/negativo
   - Basado en `tiers` de `rewards.yaml`

7. **`r_multiple_reward.py`** - Rewards por R-Multiple
   - Sistema de tramos configurables para R-Multiple positivo/negativo
   - Basado en `r_multiple_tiers` de `rewards.yaml`

8. **`leverage_reward.py`** - Rewards/Penalties por Leverage
   - Bonus por leverage alto en trades ganadores
   - Penalty por leverage alto en trades perdedores
   - Bonus por leverage conservador consistente
   - Historial de leverage para consistencia

9. **`timeframe_reward.py`** - Rewards por Timeframe
   - Bonus basado en timeframe usado para ejecución
   - Configuración de timeframes disponibles

10. **`duration_reward.py`** - Rewards por Duración
    - Penalización dura por trades de 0 barras: -2.0
    - Sin reward/penalty por 1 barra: 0.0
    - Reward por 2+ barras: +0.3
    - Bonus por duración larga

11. **`progress_bonus.py`** - Bonus por Progreso
    - Bonus por milestones de balance
    - Bonus por progreso hacia objetivo
    - Bonus por compound (crecimiento exponencial)

12. **`blocked_trade_penalty.py`** - Penalties por Trades Bloqueados
    - Penalty por trades bloqueados por diversas razones
    - Penalty por runs vacíos

### Orquestador Principal

**`reward_orchestrator.py`** - Coordina todos los sistemas
- Carga configuración desde `rewards.yaml`
- Orquesta el cálculo de todos los rewards/penalties
- Aplica clipping a los rewards finales
- Proporciona interfaz unificada

### Sistema de Selección

**`leverage_timeframe_selector.py`** - Selección dinámica de leverage y timeframe
- Calcula condiciones del mercado
- Selecciona leverage y timeframe basado en condiciones
- Integrado con el sistema de rewards

## Uso

### Importación

```python
from base_env.actions import RewardOrchestrator, LeverageTimeframeSelector
```

### Inicialización

```python
# Orquestador de rewards
reward_system = RewardOrchestrator("config/rewards.yaml")

# Selector de leverage/timeframe
selector = LeverageTimeframeSelector(
    available_leverages=[2.0, 3.0, 5.0, 10.0],
    available_timeframes=["1m", "5m"],
    symbol_config=symbol_config
)
```

### Cálculo de Rewards

```python
# Reward para un trade cerrado
trade_reward = reward_system.calculate_trade_reward(
    realized_pnl=100.0,
    notional=1000.0,
    leverage_used=3.0,
    r_multiple=1.5,
    close_reason="tp_hit",
    timeframe_used="1m",
    bars_held=5
)

# Reward completo del step
total_reward, components = reward_system.compute_reward(
    obs=obs,
    base_reward=0.0,
    events=events,
    empty_run=False,
    balance_milestones=0,
    initial_balance=1000.0,
    target_balance=1000000.0,
    steps_since_last_trade=0,
    bankruptcy_occurred=False
)
```

## Configuración

Todos los sistemas se configuran desde `config/rewards.yaml`. Cada sistema lee su configuración específica:

```yaml
# Ejemplo de configuración
tiers:
  pos: [[0.0, 1.0, 0.1], [1.0, 2.0, 0.3]]
  neg: [[0.0, 1.0, -0.2], [1.0, 2.0, -0.5]]

r_multiple_tiers:
  pos: [[0.0, 1.0, 0.1], [1.0, 2.0, 0.3]]
  neg: [[0.0, 1.0, -0.2], [1.0, 2.0, -0.5]]

leverage_rewards:
  high_leverage_bonus:
    enabled: true
    min_leverage: 5.0
    reward_per_leverage: 0.1
    max_bonus: 2.0

duration_rewards:
  enabled: true
  zero_bars_penalty: -2.0
  one_bar_reward: 0.0
  two_plus_bars_reward: 0.3
```

## Ventajas del Sistema Modular

1. **Mantenibilidad**: Cada componente está aislado y es fácil de modificar
2. **Configurabilidad**: Cada sistema puede configurarse independientemente
3. **Testabilidad**: Cada componente puede probarse por separado
4. **Extensibilidad**: Fácil agregar nuevos sistemas de rewards
5. **Debugging**: Fácil identificar qué componente causa problemas
6. **Reutilización**: Los componentes pueden usarse en diferentes contextos

## Migración

Los archivos antiguos han sido eliminados:
- `base_env/rewards/advanced_reward_system.py` → `base_env/actions/reward_orchestrator.py`
- `train_env/rewards_map.py` → `base_env/actions/` (sistemas individuales)

El sistema mantiene compatibilidad con el código existente a través del `RewardOrchestrator`.
