# app.py
"""
App de entrada. Director de orquesta que arranca el bot según configuraciones YAML centralizadas.

Usos:
  python app.py run                 # ejecuta según config/train.yaml
  python app.py run --gui           # ejecuta + abre ventana equity/balance
  python app.py gui                 # solo ventana de progreso
  python app.py config              # muestra resumen de configuraciones
"""

from __future__ import annotations
import typer
import yaml
import os
import sys
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=ALL,1=WARNING,2=ERROR,3=FATAL
os.environ["OMP_NUM_THREADS"] = "1"

app = typer.Typer(add_completion=False)

class ConfigOrchestrator:
    """Director de orquesta para configuraciones centralizadas desde YAML"""
    
    def __init__(self):
        self.symbols_config = None
        self.train_config = None
        self.risk_config = None
        self.rewards_config = None
        self.hierarchical_config = None
        self.fees_config = None
        
    def load_all_configs(self) -> Dict[str, Any]:
        """Carga todas las configuraciones YAML y las valida"""
        try:
            # Cargar symbols.yaml como source of truth
            from base_env.config.config_loader import config_loader
            self.symbols_config = config_loader.load_symbols()
            
            # Cargar train.yaml
            with open("config/train.yaml", "r", encoding="utf-8") as f:
                self.train_config = yaml.safe_load(f)
            
            # Cargar risk.yaml
            with open("config/risk.yaml", "r", encoding="utf-8") as f:
                self.risk_config = yaml.safe_load(f)
            
            # Cargar rewards.yaml
            with open("config/rewards.yaml", "r", encoding="utf-8") as f:
                self.rewards_config = yaml.safe_load(f)
            
            # Cargar hierarchical.yaml
            with open("config/hierarchical.yaml", "r", encoding="utf-8") as f:
                self.hierarchical_config = yaml.safe_load(f)
            
            # Cargar fees.yaml
            with open("config/fees.yaml", "r", encoding="utf-8") as f:
                self.fees_config = yaml.safe_load(f)
            
            return {
                "symbols": self.symbols_config,
                "train": self.train_config,
                "risk": self.risk_config,
                "rewards": self.rewards_config,
                "hierarchical": self.hierarchical_config,
                "fees": self.fees_config
            }
        except Exception as e:
            typer.echo(f"[ERROR] Error cargando configuraciones: {e}")
            raise typer.Exit(code=1)
    
    def get_symbols_for_training(self) -> List[Dict[str, Any]]:
        """Obtiene símbolos habilitados para entrenamiento desde symbols.yaml"""
        training_symbols = []
        for symbol in self.symbols_config:
            mode = f"train_{symbol.market}"
            training_symbols.append({
                "symbol": symbol.symbol,
                "mode": mode,
                "market": symbol.market,
                "leverage": {
                "min": symbol.leverage.min,
                "max": symbol.leverage.max,
                "step": symbol.leverage.step,
                "default": symbol.leverage.default
            } if symbol.leverage else None,
                "allow_shorts": symbol.allow_shorts,
                "filters": symbol.filters,
                "enabled_tfs": symbol.enabled_tfs
            })
        return training_symbols
    
    def print_config_summary(self):
        """Imprime resumen consolidado de todas las configuraciones"""
        typer.echo("\n" + "="*80)
        typer.echo("🎯 RESUMEN DE CONFIGURACIÓN DEL SISTEMA")
        typer.echo("="*80)
        
        # Símbolos
        typer.echo(f"\n📊 SÍMBOLOS CONFIGURADOS ({len(self.symbols_config)}):")
        for symbol in self.symbols_config:
            mode = f"train_{symbol.market}"
            leverage_info = ""
            if symbol.leverage:
                lev = symbol.leverage
                leverage_info = f" | Leverage: {lev.min}-{lev.max}x (step: {lev.step})"
            typer.echo(f"   • {symbol.symbol}: {mode} | TFs: {symbol.enabled_tfs}{leverage_info}")
        
        # Configuración de entrenamiento
        typer.echo(f"\n🚀 ENTRENAMIENTO:")
        typer.echo(f"   • Total timesteps: {self.train_config['ppo']['total_timesteps']:,}")
        typer.echo(f"   • N envs: {self.train_config['env']['n_envs']}")
        typer.echo(f"   • Episode length: {self.train_config['env']['episode_length']}")
        typer.echo(f"   • Warmup bars: {self.train_config['env']['warmup_bars']}")
        typer.echo(f"   • Antifreeze: {'ON' if self.train_config['env']['antifreeze']['enabled'] else 'OFF'}")
        typer.echo(f"   • Chronological: {self.train_config['env']['chronological']}")
        
        # Configuración de riesgo
        typer.echo(f"\n⚠️  RIESGO:")
        typer.echo(f"   • Bankruptcy mode: {self.risk_config['common']['bankruptcy']['mode']}")
        typer.echo(f"   • Threshold: {self.risk_config['common']['bankruptcy']['threshold_pct']}%")
        typer.echo(f"   • Spot risk: {self.risk_config['spot']['risk_pct_per_trade']}%")
        typer.echo(f"   • Futures risk: {self.risk_config['futures']['risk_pct_per_trade']}%")
        typer.echo(f"   • Force minNotional: {self.risk_config['common']['train_force_min_notional']}")
        
        # Configuración de policy
        typer.echo(f"\n🎛️  POLICY:")
        typer.echo(f"   • Min confidence: {self.hierarchical_config['gating']['min_confidence']}")
        typer.echo(f"   • Execute TFs: {self.hierarchical_config['layers']['execute_tfs']}")
        typer.echo(f"   • Confirm TFs: {self.hierarchical_config['layers']['confirm_tfs']}")
        
        # Balances y objetivos
        typer.echo(f"\n💰 BALANCES:")
        typer.echo(f"   • Initial: {self.train_config['env']['initial_balance']:,.0f} USDT")
        typer.echo(f"   • Target: {self.train_config['env']['target_balance']:,.0f} USDT")
        
        # Rangos temporales
        months_back = self.train_config['data'].get('months_back', 60)
        typer.echo(f"\n📅 DATOS:")
        typer.echo(f"   • Months back: {months_back}")
        typer.echo(f"   • TFs: {self.train_config['data']['tfs']}")
        typer.echo(f"   • Stage: {self.train_config['data']['stage']}")
        
        typer.echo("="*80)

def _load_symbols(path: str = "config/symbols.yaml") -> dict:
    """Función de compatibilidad - usar ConfigOrchestrator en su lugar"""
    orchestrator = ConfigOrchestrator()
    orchestrator.load_all_configs()
    symbols = orchestrator.get_symbols_for_training()
    return {"symbols": symbols}

def _ensure_models_dirs(models_root: str, symbol: str) -> None:
    (Path(models_root) / symbol).mkdir(parents=True, exist_ok=True)

def _import_or_exit(modname: str, attr: Optional[str] = None):
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, attr) if attr else mod
    except Exception as e:
        typer.echo(f"[ERROR] No puedo importar {modname}: {e}")
        sys.exit(1)



@app.command()
def run(
    gui: bool = typer.Option(False, help="Abrir ventana de escritorio"),
    symbol_filter: Optional[str] = typer.Option(None, help="Filtrar símbolo específico (opcional)"),
):
    """
    Arranca el bot según configuraciones YAML centralizadas.
    Entrena cronológicamente con 1 run por pasada completa del histórico.
    """
    # 1) Cargar todas las configuraciones
    orchestrator = ConfigOrchestrator()
    configs = orchestrator.load_all_configs()
    
    # 2) Obtener símbolos para entrenamiento
    training_symbols = orchestrator.get_symbols_for_training()
    
    # Filtrar símbolo específico si se especifica
    if symbol_filter:
        training_symbols = [s for s in training_symbols if s["symbol"] == symbol_filter]
        if not training_symbols:
            typer.echo(f"[ERROR] Símbolo '{symbol_filter}' no encontrado en symbols.yaml")
            raise typer.Exit(code=1)
    
    if not training_symbols:
        typer.echo("[ERROR] No hay símbolos configurados para entrenamiento")
        raise typer.Exit(code=1)

    # 3) Mostrar resumen de configuración
    orchestrator.print_config_summary()
    
    # 4) Procesar cada símbolo
    for symbol_config in training_symbols:
        symbol = symbol_config["symbol"]
        mode = symbol_config["mode"]
        market = symbol_config["market"]
        
        typer.echo(f"\n🚀 INICIANDO ENTRENAMIENTO: {symbol} ({mode})")
        typer.echo("="*60)
        
        # Crear directorio de modelos
        models_root = configs["train"]["models"]["root"]
        _ensure_models_dirs(models_root, symbol)
        
        # Abrir GUI si se solicita
        if gui:
            import subprocess, sys as _sys
            subprocess.Popen([_sys.executable, "scripts/watch_progress.py", "--symbols", symbol])
        
        # 5) Validar datos históricos
        if not _validate_historical_data(symbol, market, configs["train"]["data"]):
            typer.echo(f"[ERROR] Validación de datos fallida para {symbol}")
            continue
        
        # 6) Ejecutar entrenamiento
        try:
            _execute_training(symbol_config, configs)
        except Exception as e:
            typer.echo(f"[ERROR] Error en entrenamiento de {symbol}: {e}")
            continue

def _validate_historical_data(symbol: str, market: str, data_config: Dict[str, Any]) -> bool:
    """Valida y prepara datos históricos para el símbolo"""
    try:
        from base_env.io.parquet_loader import validate_alignment_and_gaps
        
        tfs = data_config.get("tfs", ["1m","5m","15m","1h"])
        stage = data_config.get("stage", "aligned")
        
        typer.echo(f"[DATA] Validando datos para {symbol} ({market})...")
        summary = validate_alignment_and_gaps(
            root="data", 
            symbol=symbol, 
            market=market, 
            tfs=tfs, 
            stage=stage, 
            allow_gaps=True
        )
        typer.echo(f"[DATA] ✅ Validación OK → {summary}")
        return True
        
    except Exception as e:
        typer.echo(f"[DATA] ❌ Validación fallida: {e}")
        
        # Verificar si faltan datos raw o aligned
        from pathlib import Path
        data_root = Path("data")
        
        # Verificar datos raw
        raw_path = data_root / symbol / market / "raw"
        aligned_path = data_root / symbol / market / "aligned"
        
        if not raw_path.exists() or not any(raw_path.rglob("*.parquet")):
            typer.echo(f"[DATA] ❌ No se encontraron datos RAW para {symbol} ({market})")
            typer.echo("[DATA] 📥 COMANDOS SUGERIDOS PARA DESCARGAR:")
            typer.echo(f"   python data_pipeline/scripts/download_history.py --symbol {symbol} --market {market} --tfs {','.join(tfs)} --months 36")
            typer.echo(f"   python data_pipeline/scripts/align_package.py --symbol {symbol} --market {market} --tfs {','.join(tfs)}")
            return False
        
        if not aligned_path.exists() or not any(aligned_path.rglob("*.parquet")):
            typer.echo(f"[DATA] ❌ No se encontraron datos ALIGNED para {symbol} ({market})")
            typer.echo("[DATA] 🔧 COMANDO SUGERIDO PARA ALINEAR:")
            typer.echo(f"   python data_pipeline/scripts/align_package.py --symbol {symbol} --market {market} --tfs {','.join(tfs)}")
            return False
        
        typer.echo("[DATA] 🔧 Intentando alinear datos...")
        
        # Intento de autocorrección
        try:
            import subprocess, sys as _sys
            market_cli = f"--market {market}"
            tfs_cli = ",".join(tfs)
            cmd = f"{_sys.executable} data_pipeline/scripts/align_package.py --symbol {symbol} {market_cli} --tfs {tfs_cli}"
            typer.echo(f"[DATA] Ejecutando: {cmd}")
            result = os.system(cmd)
            
            if result == 0:
                # Reintentar validación
                summary = validate_alignment_and_gaps(
                    root="data", 
                    symbol=symbol, 
                    market=market, 
                    tfs=tfs, 
                    stage=stage, 
                    allow_gaps=False
                )
                typer.echo(f"[DATA] ✅ Validación OK tras alineación → {summary}")
                return True
            else:
                typer.echo(f"[DATA] ❌ Alineación falló con código {result}")
                return False
                
        except Exception as e2:
            typer.echo(f"[DATA] ❌ Error en alineación: {e2}")
            return False

def _execute_training(symbol_config: Dict[str, Any], configs: Dict[str, Any]):
    """Ejecuta el entrenamiento para un símbolo específico"""
    symbol = symbol_config["symbol"]
    
    # Configurar variables de entorno para el entrenamiento
    os.environ["TRAINING_SYMBOL"] = symbol
    os.environ["TRAINING_MODE"] = symbol_config["mode"]
    os.environ["TRAINING_MARKET"] = symbol_config["market"]
    
    # Ejecutar entrenamiento
    from scripts.train_ppo import main as train_main
    train_main()

@app.command()
def config():
    """Muestra resumen detallado de todas las configuraciones YAML."""
    orchestrator = ConfigOrchestrator()
    orchestrator.load_all_configs()
    orchestrator.print_config_summary()

@app.command()
def gui(
    symbol: str = typer.Option("BTCUSDT", help="Símbolo a visualizar"),
):
    """Abre solo la ventana de escritorio de progreso."""
    import subprocess, sys as _sys
    subprocess.call([_sys.executable, "scripts/watch_progress.py", "--symbol", symbol])

if __name__ == "__main__":
    app()
