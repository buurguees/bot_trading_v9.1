# train_env/callbacks/main_model_saver.py
"""
Callback para guardar el modelo principal de forma segura y eficiente.

Mejoras implementadas:
- Logging estructurado en lugar de print
- Type hints completos
- Validación robusta de parámetros
- Manejo de errores específicos
- Métricas de rendimiento
- Soporte para ModelManager opcional
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict
from stable_baselines3.common.callbacks import BaseCallback
from .callback_utils import (
    CallbackLogger, validate_callback_params, safe_json_save,
    create_callback_logger, create_callback_metrics, format_timesteps
)

class MainModelSaver(BaseCallback):
    """
    Guarda el modelo principal usando ModelManager para gestión segura.
    
    Este callback guarda el modelo de entrenamiento periódicamente usando
    un ModelManager si está disponible, o el método estándar como fallback.
    
    Args:
        save_every_steps: Intervalo de steps para guardar el modelo.
        fixed_path: Ruta fija donde guardar el modelo.
        model_manager: Instancia de ModelManager para guardado seguro (opcional).
        verbose: Nivel de verbosidad (0: silent, 1: info, 2: debug).
        backup_count: Número de backups a mantener.
        compress: Si comprimir el modelo guardado.
        
    Example:
        >>> saver = MainModelSaver(
        ...     save_every_steps=10000,
        ...     fixed_path="models/BTCUSDT/main_model.zip",
        ...     model_manager=model_manager,
        ...     verbose=1
        ... )
    """
    
    def __init__(self, 
                 save_every_steps: int, 
                 fixed_path: str, 
                 model_manager: Optional[Any] = None,
                 verbose: int = 0,
                 backup_count: int = 3,
                 compress: bool = True):
        super().__init__(verbose)
        
        # Validar parámetros
        validate_callback_params(
            save_every_steps=save_every_steps,
            verbose=verbose
        )
        
        self.save_every_steps = int(save_every_steps)
        self.fixed_path = Path(fixed_path)
        self.model_manager = model_manager
        self.backup_count = max(0, int(backup_count))
        self.compress = bool(compress)
        
        # Configurar logging y métricas
        self.logger = create_callback_logger("MainModelSaver", verbose)
        self.metrics = create_callback_metrics()
        
        # Crear directorio padre
        try:
            self.fixed_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directorio de guardado configurado: {self.fixed_path.parent}")
        except Exception as e:
            self.logger.error(f"Error creando directorio: {e}")
            raise
        
        # Verificar permisos de escritura
        self._check_write_permissions()
        
        self.logger.info(f"MainModelSaver inicializado - guardando cada {format_timesteps(self.save_every_steps)} steps")
    
    def _check_write_permissions(self) -> None:
        """Verifica permisos de escritura en el directorio."""
        try:
            test_file = self.fixed_path.parent / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            self.logger.error(f"Sin permisos de escritura en {self.fixed_path.parent}: {e}")
            raise PermissionError(f"No se puede escribir en {self.fixed_path.parent}")
    
    def _on_step(self) -> bool:
        """Guarda el modelo si es el momento apropiado."""
        if self.num_timesteps % self.save_every_steps == 0:
            self._save_model()
        return True
    
    def _save_model(self) -> None:
        """Guarda el modelo de forma segura."""
        try:
            self.logger.info(f"Guardando modelo en step {format_timesteps(self.num_timesteps)}")
            
            # Crear backup si existe el archivo anterior
            self._create_backup()
            
            # Guardar modelo
            if self.model_manager:
                self._save_with_manager()
            else:
                self._save_standard()
            
            # Registrar operación exitosa
            self.metrics.record_operation()
            self.logger.info(f"Modelo guardado exitosamente en {self.fixed_path}")
            
        except Exception as e:
            self.metrics.record_error()
            self.logger.error(f"Error guardando modelo: {e}")
            # No re-lanzar excepción para no interrumpir entrenamiento
    
    def _save_with_manager(self) -> None:
        """Guarda usando ModelManager."""
        try:
            if hasattr(self.model_manager, 'ensure_safe_save'):
                self.model_manager.ensure_safe_save(self.model)
            elif hasattr(self.model_manager, 'save_model'):
                self.model_manager.save_model(self.model, str(self.fixed_path))
            else:
                self.logger.warning("ModelManager no tiene método de guardado reconocido, usando método estándar")
                self._save_standard()
        except Exception as e:
            self.logger.error(f"Error con ModelManager: {e}")
            self.logger.info("Intentando guardado estándar como fallback")
            self._save_standard()
    
    def _save_standard(self) -> None:
        """Guarda usando método estándar de SB3."""
        try:
            self.model.save(str(self.fixed_path))
        except Exception as e:
            self.logger.error(f"Error en guardado estándar: {e}")
            raise
    
    def _create_backup(self) -> None:
        """Crea backup del modelo anterior si existe."""
        if not self.fixed_path.exists() or self.backup_count <= 0:
            return
        
        try:
            # Rotar backups existentes
            for i in range(self.backup_count - 1, 0, -1):
                old_backup = self.fixed_path.with_suffix(f".backup{i}.zip")
                new_backup = self.fixed_path.with_suffix(f".backup{i + 1}.zip")
                if old_backup.exists():
                    old_backup.rename(new_backup)
            
            # Crear backup del archivo actual
            backup_path = self.fixed_path.with_suffix(".backup1.zip")
            self.fixed_path.rename(backup_path)
            
            self.logger.debug(f"Backup creado: {backup_path}")
            
        except Exception as e:
            self.logger.warning(f"Error creando backup: {e}")
    
    def _on_training_start(self) -> None:
        """Inicialización al comenzar entrenamiento."""
        self.logger.info("Iniciando guardado de modelo principal")
        self.metrics = create_callback_metrics()
    
    def _on_training_end(self) -> None:
        """Finalización al terminar entrenamiento."""
        # Guardar modelo final
        self._save_model()
        
        # Mostrar estadísticas
        stats = self.metrics.get_stats()
        self.logger.info(f"Estadísticas finales - Operaciones: {stats['operations']}, "
                        f"Errores: {stats['errors']}, Duración: {stats['duration']:.1f}s")
    
    def get_save_path(self) -> Path:
        """Retorna la ruta donde se guarda el modelo."""
        return self.fixed_path
    
    def get_backup_paths(self) -> list[Path]:
        """Retorna las rutas de los backups."""
        return [self.fixed_path.with_suffix(f".backup{i}.zip") 
                for i in range(1, self.backup_count + 1)]
    
    def cleanup_old_backups(self) -> None:
        """Limpia backups antiguos manteniendo solo los especificados."""
        try:
            for i in range(self.backup_count + 1, 10):  # Limpiar backups > backup_count
                old_backup = self.fixed_path.with_suffix(f".backup{i}.zip")
                if old_backup.exists():
                    old_backup.unlink()
                    self.logger.debug(f"Backup antiguo eliminado: {old_backup}")
        except Exception as e:
            self.logger.warning(f"Error limpiando backups antiguos: {e}")
