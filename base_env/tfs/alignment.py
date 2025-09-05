# base_env/tfs/alignment.py
"""
Verificación de alineación multi-timeframe con funcionalidades avanzadas.

Este módulo proporciona herramientas para alinear datos de múltiples timeframes,
con soporte para modos estricto/flexible, validación de calidad de datos,
logging detallado y métricas de rendimiento.
"""
from __future__ import annotations
from typing import Dict, Literal, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import time

# Configuración de logging
logger = logging.getLogger(__name__)

TF = Literal["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "1w", "1M"]

class AlignmentMode(Enum):
    """Modos de alineación de datos."""
    STRICT = "strict"          # Exige todos los TFs
    FLEXIBLE = "flexible"      # Permite TFs faltantes
    BEST_EFFORT = "best_effort"  # Usa los TFs disponibles

class DataQuality(Enum):
    """Niveles de calidad de datos."""
    EXCELLENT = "excellent"    # Todos los TFs presentes
    GOOD = "good"             # >80% TFs presentes
    FAIR = "fair"             # >50% TFs presentes
    POOR = "poor"             # <50% TFs presentes

@dataclass
class AlignmentResult:
    """Resultado de la operación de alineación."""
    data: Dict[TF, Dict[str, float]]
    quality: DataQuality
    missing_tfs: List[TF] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_valid(self) -> bool:
        """Indica si el resultado es válido para uso."""
        return self.quality in [DataQuality.EXCELLENT, DataQuality.GOOD]
    
    @property
    def coverage_percentage(self) -> float:
        """Porcentaje de cobertura de timeframes."""
        total_requested = len(self.data) + len(self.missing_tfs)
        if total_requested == 0:
            return 0.0
        return (len(self.data) / total_requested) * 100

class MTFAligner:
    """
    Alineador de múltiples timeframes con funcionalidades avanzadas.
    
    Características:
    - Múltiples modos de alineación
    - Validación de calidad de datos
    - Logging detallado
    - Métricas de rendimiento
    - Tolerancia a fallos configurables
    """
    
    # Jerarquía de timeframes (menor a mayor)
    TF_HIERARCHY: Dict[TF, int] = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "8h": 480,
        "12h": 720, "1d": 1440, "1w": 10080, "1M": 43200
    }
    
    def __init__(
        self,
        mode: AlignmentMode = AlignmentMode.STRICT,
        min_quality: DataQuality = DataQuality.GOOD,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
        enable_caching: bool = True
    ) -> None:
        """
        Inicializa el alineador MTF.
        
        Args:
            mode: Modo de alineación a usar
            min_quality: Calidad mínima requerida
            timeout_seconds: Timeout para operaciones del broker
            retry_attempts: Número de intentos en caso de fallo
            enable_caching: Habilita caché de resultados
        """
        self.mode = mode
        self.min_quality = min_quality
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.enable_caching = enable_caching
        
        # Métricas internas
        self._alignment_count = 0
        self._success_count = 0
        self._cache: Dict[str, Tuple[AlignmentResult, datetime]] = {}
        self._cache_ttl = timedelta(seconds=60)  # TTL del caché
        
        logger.info(f"MTFAligner inicializado: modo={mode.value}, calidad_min={min_quality.value}")

    def align(
        self,
        broker: Any,
        required_tfs: List[TF],
        symbol: Optional[str] = None,
        bar_time: Optional[datetime] = None
    ) -> AlignmentResult:
        """
        Alinea datos de múltiples timeframes.
        
        Args:
            broker: Instancia del broker de datos
            required_tfs: Lista de timeframes requeridos
            symbol: Símbolo para filtrar (opcional)
            bar_time: Tiempo específico para alineación (opcional)
            
        Returns:
            AlignmentResult con los datos alineados y metadatos
            
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si falla la alineación en modo estricto
            TimeoutError: Si se excede el tiempo límite
        """
        start_time = time.perf_counter()
        self._alignment_count += 1
        
        # Validación de entrada
        self._validate_inputs(required_tfs, broker)
        
        # Verificar caché
        if self.enable_caching:
            cached_result = self._get_cached_result(required_tfs, symbol, bar_time)
            if cached_result:
                logger.debug("Resultado obtenido del caché")
                return cached_result
        
        try:
            # Intentar alineación con reintentos
            result = self._perform_alignment_with_retry(
                broker, required_tfs, symbol, bar_time
            )
            
            # Calcular tiempo de ejecución
            execution_time = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            # Validar resultado según modo
            self._validate_result(result, required_tfs)
            
            # Guardar en caché
            if self.enable_caching:
                self._cache_result(result, required_tfs, symbol, bar_time)
            
            self._success_count += 1
            logger.info(
                f"Alineación exitosa: {len(result.data)}/{len(required_tfs)} TFs, "
                f"calidad={result.quality.value}, tiempo={execution_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error en alineación: {e}")
            raise

    def _validate_inputs(self, required_tfs: List[TF], broker: Any) -> None:
        """Valida los parámetros de entrada."""
        if not required_tfs:
            raise ValueError("La lista de timeframes requeridos no puede estar vacía")
        
        if not hasattr(broker, 'aligned_view'):
            raise ValueError("El broker debe implementar el método 'aligned_view'")
        
        # Validar timeframes
        invalid_tfs = [tf for tf in required_tfs if tf not in self.TF_HIERARCHY]
        if invalid_tfs:
            raise ValueError(f"Timeframes inválidos: {invalid_tfs}")

    def _perform_alignment_with_retry(
        self,
        broker: Any,
        required_tfs: List[TF],
        symbol: Optional[str],
        bar_time: Optional[datetime]
    ) -> AlignmentResult:
        """Realiza la alineación con reintentos."""
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Llamar al broker con timeout
                aligned_data = self._call_broker_with_timeout(
                    broker, required_tfs, symbol, bar_time
                )
                
                # Crear resultado
                result = self._create_alignment_result(aligned_data, required_tfs)
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Backoff exponencial
                    logger.warning(f"Intento {attempt + 1} falló, reintentando en {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Todos los intentos fallaron")
        
        raise last_exception or RuntimeError("Alineación fallida después de todos los reintentos")

    def _call_broker_with_timeout(
        self,
        broker: Any,
        required_tfs: List[TF],
        symbol: Optional[str],
        bar_time: Optional[datetime]
    ) -> Dict[TF, Dict[str, float]]:
        """Llama al broker con timeout."""
        # En una implementación real, aquí se implementaría el timeout
        # Por ahora, simplemente llamamos al método
        return broker.aligned_view(required_tfs)

    def _create_alignment_result(
        self,
        aligned_data: Dict[TF, Dict[str, float]],
        required_tfs: List[TF]
    ) -> AlignmentResult:
        """Crea el resultado de alineación con metadatos."""
        available_tfs = set(aligned_data.keys())
        required_tfs_set = set(required_tfs)
        missing_tfs = list(required_tfs_set - available_tfs)
        
        # Calcular calidad
        quality = self._calculate_data_quality(len(aligned_data), len(required_tfs))
        
        # Generar warnings
        warnings = []
        if missing_tfs:
            warnings.append(f"Timeframes faltantes: {missing_tfs}")
        
        # Validar consistencia de datos
        data_warnings = self._validate_data_consistency(aligned_data)
        warnings.extend(data_warnings)
        
        return AlignmentResult(
            data=aligned_data,
            quality=quality,
            missing_tfs=missing_tfs,
            warnings=warnings
        )

    def _calculate_data_quality(self, available_count: int, required_count: int) -> DataQuality:
        """Calcula la calidad de los datos basada en cobertura."""
        if required_count == 0:
            return DataQuality.POOR
        
        coverage = available_count / required_count
        
        if coverage == 1.0:
            return DataQuality.EXCELLENT
        elif coverage >= 0.8:
            return DataQuality.GOOD
        elif coverage >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    def _validate_data_consistency(
        self,
        aligned_data: Dict[TF, Dict[str, float]]
    ) -> List[str]:
        """Valida la consistencia de los datos alineados."""
        warnings = []
        
        # Verificar que todos los TFs tengan las mismas claves
        if aligned_data:
            first_keys = set(next(iter(aligned_data.values())).keys())
            for tf, data in aligned_data.items():
                if set(data.keys()) != first_keys:
                    warnings.append(f"TF {tf} tiene claves inconsistentes")
        
        return warnings

    def _validate_result(self, result: AlignmentResult, required_tfs: List[TF]) -> None:
        """Valida el resultado según el modo de alineación."""
        if self.mode == AlignmentMode.STRICT and result.missing_tfs:
            raise RuntimeError(f"Modo estricto: faltan TFs {result.missing_tfs}")
        
        if result.quality.value < self.min_quality.value:
            if self.mode == AlignmentMode.STRICT:
                raise RuntimeError(f"Calidad insuficiente: {result.quality.value} < {self.min_quality.value}")
            else:
                logger.warning(f"Calidad por debajo del mínimo: {result.quality.value}")

    def _get_cached_result(
        self,
        required_tfs: List[TF],
        symbol: Optional[str],
        bar_time: Optional[datetime]
    ) -> Optional[AlignmentResult]:
        """Obtiene resultado del caché si está disponible y válido."""
        cache_key = self._generate_cache_key(required_tfs, symbol, bar_time)
        
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return result
            else:
                # Remover entrada expirada
                del self._cache[cache_key]
        
        return None

    def _cache_result(
        self,
        result: AlignmentResult,
        required_tfs: List[TF],
        symbol: Optional[str],
        bar_time: Optional[datetime]
    ) -> None:
        """Guarda resultado en caché."""
        cache_key = self._generate_cache_key(required_tfs, symbol, bar_time)
        self._cache[cache_key] = (result, datetime.now())

    def _generate_cache_key(
        self,
        required_tfs: List[TF],
        symbol: Optional[str],
        bar_time: Optional[datetime]
    ) -> str:
        """Genera clave única para caché."""
        tfs_str = ",".join(sorted(required_tfs))
        symbol_str = symbol or "ALL"
        time_str = bar_time.isoformat() if bar_time else "CURRENT"
        return f"{tfs_str}:{symbol_str}:{time_str}"

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del alineador."""
        success_rate = (self._success_count / self._alignment_count * 100) if self._alignment_count > 0 else 0
        
        return {
            "total_alignments": self._alignment_count,
            "successful_alignments": self._success_count,
            "success_rate_percent": round(success_rate, 2),
            "cache_entries": len(self._cache),
            "mode": self.mode.value,
            "min_quality": self.min_quality.value
        }

    def clear_cache(self) -> None:
        """Limpia el caché de resultados."""
        self._cache.clear()
        logger.info("Caché limpiado")

    def set_mode(self, mode: AlignmentMode) -> None:
        """Cambia el modo de alineación."""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Modo cambiado de {old_mode.value} a {mode.value}")

    @classmethod
    def get_tf_duration_minutes(cls, tf: TF) -> int:
        """Obtiene la duración en minutos de un timeframe."""
        return cls.TF_HIERARCHY.get(tf, 0)

    @classmethod
    def sort_timeframes(cls, tfs: List[TF]) -> List[TF]:
        """Ordena timeframes de menor a mayor duración."""
        return sorted(tfs, key=lambda x: cls.TF_HIERARCHY.get(x, 0))