# train_env/callbacks/main_model_saver.py
from __future__ import annotations
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

class MainModelSaver(BaseCallback):
    """
    Guarda el modelo principal usando ModelManager para gestión segura
    """
    def __init__(self, save_every_steps: int, fixed_path: str, model_manager=None, verbose: int = 0):
        super().__init__(verbose)
        self.save_every_steps = int(save_every_steps)
        self.fixed_path = Path(fixed_path)
        self.fixed_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_every_steps == 0:
            if self.verbose: print(f"[MODEL] Saving to {self.fixed_path}")
            
            if self.model_manager:
                # Usar ModelManager para guardado seguro
                self.model_manager.ensure_safe_save(self.model)
            else:
                # Fallback al método original
                self.model.save(str(self.fixed_path))
        return True
