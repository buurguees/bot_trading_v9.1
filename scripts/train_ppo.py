# scripts/train_ppo.py
from __future__ import annotations
import os, yaml, numpy as np, sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from train_env.vec_factory_chrono import make_vec_envs_chrono
from train_env.callbacks import PeriodicCheckpoint, StrategyKeeper, StrategyConsultant, AntiBadStrategy, MainModelSaver
from train_env.learning_rate_reset_callback import LearningRateResetCallback
from train_env.strategy_aggregator import aggregate_top_k
from base_env.config.symbols_loader import load_symbols

def main():
    with open("config/train.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"]); np.random.seed(seed)
    
    # Cargar símbolos desde symbols.yaml
    syms = load_symbols("config/symbols.yaml")
    sym0 = next((s for s in syms if s.mode.startswith("train")), syms[0])
    symbol_cfg = {
        "symbol": sym0.symbol,
        "mode": sym0.mode,
        "leverage": (sym0.leverage.model_dump() if sym0.leverage else None)
    }
    
    model_dir = os.path.join(cfg["models"]["root"], sym0.symbol)
    os.makedirs(model_dir, exist_ok=True)

    # Rutas de artefactos por símbolo
    main_model_path = os.path.join(model_dir, f"{sym0.symbol}_PPO.zip")
    strat_prov = os.path.join(model_dir, f"{sym0.symbol}_strategies_provisional.jsonl")
    strat_best = os.path.join(model_dir, f"{sym0.symbol}_strategies.json")
    strat_bad = os.path.join(model_dir, f"{sym0.symbol}_bad_strategies.json")  # ← NUEVO: Estrategias malas

    # Vec envs (cronológicos)
    venv = make_vec_envs_chrono(
        n_envs=cfg["env"]["n_envs"], seed=seed,
        data_cfg=cfg["data"], env_cfg=cfg["env"],
        logging_cfg=cfg["logging"], models_cfg=cfg["models"],
        symbol_cfg=symbol_cfg              # <— aquí va el YAML del símbolo
    )

    # Logger
    log_dir = cfg["log_dir"]; os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout","tensorboard"])

    # PPO (crear o reanudar)
    ppo_cfg = cfg["ppo"]
    overwrite = cfg["models"].get("overwrite", False)
    
    if os.path.exists(main_model_path) and overwrite:
        print(f"[MODEL] Resuming from {main_model_path}")
        model = PPO.load(main_model_path, env=venv, device="auto", print_system_info=True)
    elif os.path.exists(main_model_path) and not overwrite:
        print(f"[MODEL] Model exists but overwrite=False, creating new model")
        print(f"[MODEL] Existing model will be backed up to {main_model_path}.backup")
        # Hacer backup del modelo existente
        import shutil
        backup_path = f"{main_model_path}.backup"
        shutil.copy2(main_model_path, backup_path)
        print(f"[MODEL] Backup created: {backup_path}")
        # Continuar con la creación del modelo nuevo (sin else)
    
    # Crear modelo nuevo si no existe o si overwrite=False
    if not os.path.exists(main_model_path) or (os.path.exists(main_model_path) and not overwrite):
        # ← NUEVO: Configuración mejorada para evitar bloqueo del agente
        policy_kwargs = ppo_cfg.get("policy_kwargs", {})
        if policy_kwargs:
            # Convertir activation_fn string a función real
            if policy_kwargs.get("activation_fn") == "tanh":
                import torch.nn as nn
                policy_kwargs["activation_fn"] = nn.Tanh
        
        model = PPO(
            "MlpPolicy", venv,
            n_steps=ppo_cfg["n_steps"], batch_size=ppo_cfg["batch_size"],
            learning_rate=ppo_cfg["learning_rate"], gamma=ppo_cfg["gamma"], gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"], ent_coef=ppo_cfg["ent_coef"], vf_coef=ppo_cfg["vf_coef"],
            n_epochs=ppo_cfg["n_epochs"], seed=seed,
            tensorboard_log=ppo_cfg.get("tensorboard_log", None),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            target_kl=ppo_cfg.get("target_kl", 0.01),
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    model.set_logger(new_logger)

    # Callbacks
    ckpt_every = int(cfg["logging"]["checkpoint_every_steps"])
    callbacks = [
        PeriodicCheckpoint(save_every_steps=ckpt_every, save_path=os.path.join(log_dir,"checkpoints"), verbose=1),
        StrategyKeeper(
            provisional_file=strat_prov,
            best_json_file=strat_best,
            top_k=int(cfg["logging"].get("top_k", 1000)),
            every_steps=ckpt_every,
            verbose=1
        ),
        # ← NUEVO: Callback que consulta estrategias existentes para mejorar el aprendizaje
        StrategyConsultant(
            strategies_file=strat_best,
            consult_every_steps=ckpt_every,  # Consultar cada checkpoint
            verbose=1
        ),
        # ← NUEVO: Callback que identifica y evita las PEORES estrategias
        AntiBadStrategy(
            strategies_file=strat_best,
            bad_strategies_file=strat_bad,
            consult_every_steps=ckpt_every,  # Consultar cada checkpoint
            verbose=1
        ),
        # ← NUEVO: Callback para reset automático del learning rate (condicional por YAML)
        MainModelSaver(save_every_steps=int(cfg["models"]["save_every_steps"]), fixed_path=main_model_path, verbose=1),
    ]

    # ← NUEVO: Sistema anti-congelamiento (mensajes según YAML)
    print(f"🛡️ Sistema anti-congelamiento activado:")
    print(f"   - Entropy coefficient: {ppo_cfg['ent_coef']}")
    print(f"   - Learning rate annealing: {'Activado' if ppo_cfg.get('anneal_lr', False) else 'Desactivado'}")
    print(f"   - Target KL: {ppo_cfg.get('target_kl', 'No configurado')}")
    lr_reset_cfg = ppo_cfg.get('lr_reset', {"enabled": False})
    if lr_reset_cfg.get('enabled', False):
        print(f"   - Learning rate reset: {lr_reset_cfg.get('threshold_runs', 30)} runs vacíos → LR variable (1e-4 a 1e-2)")
    else:
        print(f"   - Learning rate reset: Desactivado")
    
    # Entrenar con manejo de errores robusto
    try:
        print(f"🚀 INICIANDO ENTRENAMIENTO PPO - {ppo_cfg['total_timesteps']:,} steps")
        print(f"📊 Configuración: {ppo_cfg['n_steps']} steps/env, {ppo_cfg['batch_size']} batch, {ppo_cfg['learning_rate']} lr")
        
        model.learn(total_timesteps=int(ppo_cfg["total_timesteps"]), callback=callbacks)
        
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
