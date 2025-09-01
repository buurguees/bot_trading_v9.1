# flow.md
# Descripción: Diagrama de flujo del entorno base en Mermaid (visible en GitHub/VS Code con extensión Mermaid).
# Ubicación: base_env/docs/flow.md

# Flujo BaseEnv (simplificado)

```mermaid
flowchart TD
    A[config/*.yaml<br/>settings/symbols/risk/fees/pipeline/hierarchical/oms] --> B[io/DataBroker]
    B --> C[tfs/ Alineación MTF<br/>(1d,4h → 1h,15m → 5m,1m)]
    C --> D[features/ + smc/<br/>Indicadores + Estructura/Zonas]
    D --> E[analysis/<br/>Jerárquico: señales + confidence]
    E --> F[policy/<br/>Gating + Confluencias → Acción]
    F --> G[risk/<br/>Sizing, exposición, lev≤3x, breakers]
    G --> H[accounting/<br/>Balances, Fees, PnL R/UR, MFE/MAE, DD]
    H --> I[events/<br/>OrderOpened/Closed, SL/TP, Breakers, ...]
    H --> J[[Dashboard/Logs]]
    F --> K[OMS Adapter<br/>(Sim/Paper/Live)]
