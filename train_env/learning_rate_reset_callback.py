# train_env/learning_rate_reset_callback.py
# Descripción: Callback para reset automático del learning rate cuando hay runs vacíos consecutivos

from typing import Any, Dict, List, Optional, Union
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class LearningRateResetCallback(BaseCallback):
    """
    Callback que detecta runs vacíos consecutivos y resetea el learning rate
    para forzar al agente a explorar nuevas estrategias.
    """
    
    def __init__(
        self,
        env: VecEnv,
        reset_threshold: int = 30,  # ← NUEVO: Umbral más bajo para activación más temprana
        new_learning_rate: float = 1e-3,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.env = env
        self.reset_threshold = reset_threshold
        self.new_learning_rate = new_learning_rate
        self.last_reset_step = 0
        self.reset_count = 0
        self.cooldown_steps = 5000  # ← NUEVO: Cooldown más largo para evitar resets excesivos
        self.learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 5e-3]  # ← NUEVO: Más opciones de LR
        self.entropy_boost = 0.2  # ← NUEVO: Boost temporal de entropía
        self.entropy_decay_steps = 10000  # ← NUEVO: Pasos para reducir entropía gradualmente
        self.original_ent_coef = None  # ← NUEVO: Entropía original para restaurar
        
    def _on_step(self) -> bool:
        """Se ejecuta en cada step del entrenamiento"""
        
        # Verificar si algún entorno necesita reset de learning rate
        needs_reset = False
        
        # Acceder a los entornos de manera compatible con SubprocVecEnv
        try:
            # Para SubprocVecEnv, usar get_attr para acceder a métodos
            if hasattr(self.env, 'get_attr'):
                reset_flags = self.env.get_attr('needs_learning_rate_reset')
                if any(reset_flags):
                    needs_reset = True
            else:
                # Fallback para otros tipos de VecEnv
                for i in range(self.env.num_envs):
                    if hasattr(self.env.envs[i], 'needs_learning_rate_reset'):
                        if self.env.envs[i].needs_learning_rate_reset():
                            needs_reset = True
                            break
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Error accediendo a entornos: {e}")
            return True
        
        if needs_reset:
            # ← NUEVO: Verificar cooldown para evitar resets excesivos
            steps_since_last_reset = self.num_timesteps - self.last_reset_step
            if steps_since_last_reset < self.cooldown_steps:
                if self.verbose > 0:
                    print(f"⏳ Cooldown activo: {steps_since_last_reset}/{self.cooldown_steps} steps")
                return True
            
            # ← NUEVO: Seleccionar learning rate diferente y boost de entropía
            old_lr = self.model.learning_rate
            old_ent_coef = self.model.ent_coef
            
            # Guardar entropía original si es la primera vez
            if self.original_ent_coef is None:
                self.original_ent_coef = old_ent_coef
            
            new_lr = self.learning_rates[self.reset_count % len(self.learning_rates)]
            new_ent_coef = old_ent_coef + self.entropy_boost  # Boost temporal de entropía
            
            self.model.learning_rate = new_lr
            self.model.ent_coef = new_ent_coef
            
            # Reset de flags en todos los entornos
            try:
                if hasattr(self.env, 'get_attr'):
                    # Para SubprocVecEnv
                    self.env.env_method('reset_learning_rate_flag')
                else:
                    # Fallback para otros tipos
                    for i in range(self.env.num_envs):
                        if hasattr(self.env.envs[i], 'reset_learning_rate_flag'):
                            self.env.envs[i].reset_learning_rate_flag()
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️ Error reseteando flags: {e}")
            
            # Logging
            self.reset_count += 1
            self.last_reset_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"🔄 LEARNING RATE RESET #{self.reset_count}")
                print(f"   Step: {self.num_timesteps}")
                print(f"   LR anterior: {old_lr:.2e} → LR nuevo: {new_lr:.2e}")
                print(f"   Entropía anterior: {old_ent_coef:.3f} → Entropía nueva: {new_ent_coef:.3f}")
                print(f"   Motivo: {self.reset_threshold} runs vacíos consecutivos")
                print(f"   Cooldown: {self.cooldown_steps} steps")
                print(f"   El agente ahora explorará nuevas estrategias con mayor entropía")
        
        # ← NUEVO: Reducir entropía gradualmente después del boost
        if self.original_ent_coef is not None and self.model.ent_coef > self.original_ent_coef:
            steps_since_boost = self.num_timesteps - self.last_reset_step
            if steps_since_boost > 0 and steps_since_boost <= self.entropy_decay_steps:
                # Reducir entropía gradualmente
                decay_factor = 1.0 - (steps_since_boost / self.entropy_decay_steps)
                target_ent_coef = self.original_ent_coef + (self.entropy_boost * decay_factor)
                self.model.ent_coef = max(target_ent_coef, self.original_ent_coef)
        
        return True
    
    def _on_training_end(self) -> None:
        """Se ejecuta al final del entrenamiento"""
        if self.verbose > 0:
            print(f"📊 Resumen de Learning Rate Resets:")
            print(f"   Total resets: {self.reset_count}")
            print(f"   Último reset en step: {self.last_reset_step}")
            print(f"   LR final: {self.model.learning_rate:.2e}")
