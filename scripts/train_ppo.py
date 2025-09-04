# scripts/train_ppo.py
from __future__ import annotations
import os, yaml, numpy as np, sys
from pathlib import Path
import multiprocessing as mp

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ← NUEVO: Configurar multiprocessing para Windows
if __name__ == "__main__":
    # Configurar el método de inicio de procesos para Windows
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Ya está configurado
        pass

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from train_env.vec_factory_chrono import make_vec_envs_chrono
from train_env.callbacks import PeriodicCheckpoint, StrategyKeeper, StrategyConsultant, AntiBadStrategy, MainModelSaver
from train_env.callbacks import TrainingMetricsCallback
from train_env.learning_rate_reset_callback import LearningRateResetCallback
from train_env.strategy_aggregator import aggregate_top_k
from train_env.model_manager import ModelManager
from base_env.config.config_loader import config_loader

def main():
    with open("config/train.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"]); np.random.seed(seed)
    
    # Cargar símbolos desde symbols.yaml
    syms = config_loader.load_symbols()
    # ← NUEVO: Priorizar futuros para train_futures
    sym0 = next((s for s in syms if s.market == "futures"), None)
    if sym0 is None:
        sym0 = next((s for s in syms if s.market == "spot"), syms[0])
    symbol_cfg = {
        "symbol": sym0.symbol,
        "mode": f"train_{sym0.market}",
        "leverage": (sym0.leverage.__dict__ if sym0.leverage else None)
    }
    
    # Gestor de modelos por símbolo
    model_manager = ModelManager(
        symbol=sym0.symbol,
        models_root=cfg["models"]["root"],
        overwrite=cfg["models"]["overwrite"]
    )
    
    # Mostrar resumen del modelo
    model_manager.print_summary()
    
    # Obtener rutas de archivos
    file_paths = model_manager.get_file_paths()
    main_model_path = str(file_paths["model"])
    strat_prov = str(file_paths["provisional"])
    strat_best = str(file_paths["strategies"])
    strat_bad = str(file_paths["bad_strategies"])

    # ← NUEVO: Configurar verbosidad de logging
    train_verbosity = cfg["logging"].get("train_verbosity", "low")
    verbosity_levels = {
        "low": {"print_interval": 1000, "log_interval": 1000, "verbose": 0},
        "medium": {"print_interval": 100, "log_interval": 100, "verbose": 1},
        "high": {"print_interval": 1, "log_interval": 1, "verbose": 2}
    }
    
    verbosity_config = verbosity_levels.get(train_verbosity, verbosity_levels["low"])
    print(f"📊 Configuración de logging: {train_verbosity} (intervalo: {verbosity_config['print_interval']} steps)")

    # Vec envs (cronológicos)
    venv = make_vec_envs_chrono(
        n_envs=cfg["env"]["n_envs"], seed=seed,
        data_cfg=cfg["data"], env_cfg=cfg["env"],
        logging_cfg=cfg["logging"], models_cfg=cfg["models"],
        symbol_cfg=symbol_cfg,              # <— aquí va el YAML del símbolo
        runs_log_cfg=cfg.get("runs_log", {})  # <— configuración de retención de runs
    )

    # Logger
    log_dir = cfg["log_dir"]; os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout","tensorboard"])

    # PPO (crear o reanudar usando ModelManager)
    ppo_cfg = cfg["ppo"]
    
    # Configurar parámetros de PPO
    policy_kwargs = ppo_cfg.get("policy_kwargs", {})
    if policy_kwargs:
        # Convertir activation_fn string a función real
        if policy_kwargs.get("activation_fn") == "tanh":
            import torch.nn as nn
            policy_kwargs["activation_fn"] = nn.Tanh
    
    # ← NUEVO: Configurar learning rate annealing
    if ppo_cfg.get("anneal_lr", False):
        # Learning rate annealing: 3e-4 → 1e-5
        import torch.optim as optim
        def lr_schedule(progress_remaining):
            return 1e-5 + (3e-4 - 1e-5) * progress_remaining
        ppo_cfg["learning_rate"] = lr_schedule
    
    # Separar total_timesteps del resto de la configuración PPO
    total_timesteps = ppo_cfg.pop("total_timesteps", 50000000)
    
    # Filtrar parámetros que no son válidos para PPO
    ppo_valid_params = {
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
        'clip_range', 'clip_range_vf', 'ent_coef', 'vf_coef', 'max_grad_norm',
        'use_sde', 'sde_sample_freq', 'target_kl', 'tensorboard_log', 'policy_kwargs',
        'verbose', 'seed', 'device', 'stats_window_size', 'tensorboard_log'
    }
    
    # Crear diccionario solo con parámetros válidos para PPO
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k in ppo_valid_params}
    
    # ← NUEVO: Aplicar verbosidad al modelo PPO
    ppo_kwargs["verbose"] = verbosity_config["verbose"]
    
    # Cargar o crear modelo usando ModelManager
    model = model_manager.load_model(
        env=venv,
        device="auto",
        **ppo_kwargs
    )
    model.set_logger(new_logger)

    # Callbacks
    ckpt_every = int(cfg["logging"]["checkpoint_every_steps"])
    callback_verbose = 1 if train_verbosity in ["medium", "high"] else 0
    
    callbacks = [
        PeriodicCheckpoint(save_every_steps=ckpt_every, save_path=os.path.join(log_dir,"checkpoints"), verbose=callback_verbose),
        StrategyKeeper(
            provisional_file=strat_prov,
            best_json_file=strat_best,
            top_k=int(cfg["logging"].get("top_k", 1000)),
            every_steps=ckpt_every,
            verbose=callback_verbose
        ),
        # ← NUEVO: Callback que consulta estrategias existentes para mejorar el aprendizaje
        StrategyConsultant(
            strategies_file=strat_best,
            consult_every_steps=ckpt_every,  # Consultar cada checkpoint
            verbose=callback_verbose
        ),
        # ← NUEVO: Callback que identifica y evita las PEORES estrategias
        AntiBadStrategy(
            strategies_file=strat_best,
            bad_strategies_file=strat_bad,
            consult_every_steps=ckpt_every,  # Consultar cada checkpoint
            verbose=callback_verbose
        ),
        # ← NUEVO: Callback para guardar modelo principal usando ModelManager
        MainModelSaver(save_every_steps=int(cfg["models"]["save_every_steps"]), fixed_path=main_model_path, model_manager=model_manager, verbose=callback_verbose),
    ]
    
    # ← NUEVO: Añadir callback de métricas de entrenamiento si está habilitado
    metrics_cfg = cfg.get("metrics", {})
    if metrics_cfg.get("enable", True):
        # Ajustar intervalo según verbosidad
        base_interval = int(metrics_cfg.get("interval", 2048))
        if train_verbosity == "low":
            metrics_interval = max(base_interval, 1000)  # Mínimo 1000 steps
        elif train_verbosity == "medium":
            metrics_interval = max(base_interval, 100)   # Mínimo 100 steps
        else:  # high
            metrics_interval = base_interval              # Usar intervalo base
        
        metrics_path_pattern = metrics_cfg.get("path_pattern", "models/{symbol}/{symbol}_train_metrics.jsonl")
        metrics_path = metrics_path_pattern.format(symbol=symbol_cfg["symbol"])
        
        metrics_callback = TrainingMetricsCallback(
            symbol=symbol_cfg["symbol"],
            mode=symbol_cfg["mode"],
            metrics_path=metrics_path,
            log_interval=metrics_interval,
            verbose=callback_verbose
        )
        callbacks.append(metrics_callback)
        print(f"📊 Callback de métricas habilitado: {metrics_path} (intervalo: {metrics_interval})")
    else:
        print("📊 Callback de métricas deshabilitado")

    # ← NUEVO: Sistema anti-congelamiento (mensajes según YAML)
    print(f"🛡️ Sistema anti-congelamiento activado:")
    print(f"   - Entropy coefficient: {ppo_cfg['ent_coef']}")
    print(f"   - Learning rate annealing: {'Activado' if ppo_cfg.get('anneal_lr', False) else 'Desactivado'}")
    print(f"   - Target KL: {ppo_cfg.get('target_kl', 'No configurado')}")
    lr_reset_cfg = ppo_cfg.get('lr_reset', {"enabled": False})
    if lr_reset_cfg.get('enabled', False):
        print(f"   - Learning rate reset: {lr_reset_cfg.get('threshold_runs', 30)} runs vacíos → LR variable (1e-4 a 1e-2)")
        # Añadir el callback de reset de learning rate
        callbacks.append(LearningRateResetCallback(
            env=venv,
            reset_threshold=lr_reset_cfg.get('threshold_runs', 30),
            verbose=1
        ))
    else:
        print(f"   - Learning rate reset: Desactivado")
    
    # Entrenar con manejo de errores robusto
    try:
        print(f"🚀 INICIANDO ENTRENAMIENTO PPO - {total_timesteps:,} steps")
        print(f"📊 Configuración: {ppo_cfg['n_steps']} steps/env, {ppo_cfg['batch_size']} batch, {ppo_cfg['learning_rate']} lr")
        print(f"🔧 Multiprocessing: {cfg['env']['n_envs']} workers con método 'spawn'")
        print(f"📝 Logging: {train_verbosity} (verbose={verbosity_config['verbose']}, intervalo={verbosity_config['print_interval']} steps)")
        
        # ← NUEVO: Configurar variables de entorno para estabilidad
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['OMP_NUM_THREADS'] = '1'  # Evitar conflictos de threading
        
        model.learn(total_timesteps=int(total_timesteps), callback=callbacks)
        
        print("✅ Entrenamiento completado exitosamente")
        
        # Guardado final en la ruta fija (sobrescribe)
        model.save(main_model_path)
        print(f"[MODEL] Final PPO saved -> {main_model_path}")

        # Última consolidación de estrategias
        aggregate_top_k(strat_prov, strat_best, int(cfg["logging"].get("top_k", 1000)))
        print(f"[STRAT] Best strategies -> {strat_best}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
        print("💾 Guardando modelo en estado actual...")
        model.save(main_model_path)
        print(f"[MODEL] Modelo guardado en estado interrumpido -> {main_model_path}")
        
    except (EOFError, BrokenPipeError) as e:
        print(f"\n❌ ERROR DE MULTIPROCESSING: {type(e).__name__}: {e}")
        print("🔧 Esto puede ser causado por:")
        print("   - Demasiados workers para los recursos disponibles")
        print("   - Problemas de comunicación entre procesos")
        print("   - Memoria insuficiente")
        print("💾 Intentando guardar modelo en estado de error...")
        try:
            model.save(main_model_path)
            print(f"[MODEL] Modelo guardado en estado de error -> {main_model_path}")
        except Exception as save_error:
            print(f"❌ No se pudo guardar el modelo: {save_error}")
        
        # No re-raise para evitar crash completo
        print("⚠️ Continuando sin re-lanzar el error...")
        
    except Exception as e:
        print(f"\n❌ ERROR durante el entrenamiento: {type(e).__name__}: {e}")
        print("💾 Intentando guardar modelo en estado de error...")
        try:
            model.save(main_model_path)
            print(f"[MODEL] Modelo guardado en estado de error -> {main_model_path}")
        except Exception as save_error:
            print(f"❌ No se pudo guardar el modelo: {save_error}")
        
        # Re-raise para debugging
        raise

if __name__ == "__main__":
    main()
