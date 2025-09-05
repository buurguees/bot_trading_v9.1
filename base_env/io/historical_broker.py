# base_env/io/historical_broker.py
# Descripción:
#   Broker histórico que consume Parquet "aligned" por TF y entrega:
#   - now_ts()
#   - next()
#   - get_price() (close del TF base)
#   - get_bar(tf) (última barra cerrada ≤ now_ts para ese TF)
#   - aligned_view(required_tfs)
#
#   Avanza sobre la secuencia de timestamps del TF base.
#   Para TFs superiores, selecciona la última barra con ts <= now_ts.
#
# Ubicación: base_env/io/historical_broker.py

from __future__ import annotations
import logging
from typing import Dict, Optional, Literal, List, Any
from pathlib import Path
from bisect import bisect_right

from .parquet_loader import load_window
from ..tfs.calendar import TF, tf_to_ms

logger = logging.getLogger(__name__)

class HistoricalBrokerError(Exception):
    """Excepción base para errores del broker histórico"""
    pass

class NoDataError(HistoricalBrokerError):
    """Error cuando no hay datos disponibles"""
    pass

class InvalidRangeError(HistoricalBrokerError):
    """Error cuando el rango de fechas es inválido"""
    pass

class ParquetHistoricalBroker:
    """
    Broker histórico que proporciona datos OHLCV desde archivos Parquet.
    
    Maneja múltiples timeframes y proporciona una interfaz unificada
    para acceder a datos históricos alineados temporalmente.
    """
    
    def __init__(
        self,
        data_root: str | Path,
        symbol: str,
        market: Literal["spot", "futures"],
        tfs: List[TF],
        base_tf: TF = "1m",
        ts_from: Optional[int] = None,
        ts_to: Optional[int] = None,
        stage: str = "aligned",
        warmup_bars: int = 5000,
    ) -> None:
        """
        Inicializa el broker histórico.
        
        Args:
            data_root: Directorio raíz de los datos
            symbol: Símbolo del instrumento (ej: "BTCUSDT")
            market: Tipo de mercado ("spot" o "futures")
            tfs: Lista de timeframes a cargar
            base_tf: Timeframe base para el timeline principal
            ts_from: Timestamp de inicio (opcional)
            ts_to: Timestamp de fin (opcional)
            stage: Etapa de procesamiento de datos
            warmup_bars: Número de barras para precalentar si no hay rango
        
        Raises:
            NoDataError: Si no hay datos para el TF base
            InvalidRangeError: Si el rango de fechas es inválido
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones de entrada
        self._validate_inputs(data_root, symbol, market, tfs, base_tf, ts_from, ts_to, warmup_bars)
        
        self.root = Path(data_root)
        self.symbol = symbol.upper()
        self.market = market
        self.tfs = list(set(tfs))  # Eliminar duplicados
        self.base_tf = base_tf
        self.ts_from = ts_from
        self.ts_to = ts_to
        self.stage = stage
        self.warmup_bars = max(1, warmup_bars)  # Asegurar al menos 1

        # Asegurar que base_tf esté en la lista de TFs
        if base_tf not in self.tfs:
            self.tfs.append(base_tf)
            logger.warning(f"TF base {base_tf} agregado a la lista de TFs")

        # Cargar datos
        self.series_by_tf: Dict[TF, Dict[int, Dict[str, float]]] = {}
        self.ts_index_by_tf: Dict[TF, List[int]] = {}
        
        self._load_data()
        self._setup_timeline()
        
        logger.info(f"Broker inicializado: {symbol}/{market}, TFs: {self.tfs}, "
                   f"rango: {self.range_ts}, barras base: {len(self._base_ts_list)}")

    def _validate_inputs(
        self, 
        data_root: str | Path, 
        symbol: str, 
        market: str, 
        tfs: List[TF], 
        base_tf: TF,
        ts_from: Optional[int], 
        ts_to: Optional[int], 
        warmup_bars: int
    ) -> None:
        """Valida los parámetros de entrada"""
        if not data_root:
            raise ValueError("data_root no puede estar vacío")
        if not symbol or not symbol.strip():
            raise ValueError("symbol no puede estar vacío")
        if market not in ["spot", "futures"]:
            raise ValueError("market debe ser 'spot' o 'futures'")
        if not tfs:
            raise ValueError("tfs no puede estar vacío")
        if warmup_bars < 1:
            raise ValueError("warmup_bars debe ser al menos 1")
        
        # Validar rango de fechas
        if ts_from is not None and ts_to is not None:
            if ts_from >= ts_to:
                raise InvalidRangeError(f"ts_from ({ts_from}) debe ser menor que ts_to ({ts_to})")

    def _load_data(self) -> None:
        """Carga los datos para todos los timeframes"""
        for tf in self.tfs:
            try:
                series = load_window(
                    self.root, self.symbol, self.market, tf, 
                    ts_from=self.ts_from, ts_to=self.ts_to, stage=self.stage
                )
                
                # Si no hay datos en el rango, intentar cargar datos históricos
                if not series and (self.ts_from is not None or self.ts_to is not None):
                    logger.warning(f"No hay datos en el rango para {tf}, cargando últimos {self.warmup_bars}")
                    series = load_window(
                        self.root, self.symbol, self.market, tf, 
                        ts_from=None, ts_to=None, stage=self.stage, 
                        limit=self.warmup_bars
                    )
                
                if not series:
                    logger.warning(f"No hay datos disponibles para TF {tf}")
                    self.series_by_tf[tf] = {}
                    self.ts_index_by_tf[tf] = []
                else:
                    # Validar y limpiar datos
                    cleaned_series = self._clean_series_data(series, tf)
                    ordered_ts = sorted(cleaned_series.keys())
                    self.series_by_tf[tf] = cleaned_series
                    self.ts_index_by_tf[tf] = ordered_ts
                    logger.debug(f"Cargado TF {tf}: {len(ordered_ts)} barras")
                    
            except Exception as e:
                logger.error(f"Error cargando datos para TF {tf}: {e}")
                self.series_by_tf[tf] = {}
                self.ts_index_by_tf[tf] = []

    def _clean_series_data(self, series: Dict[int, Dict[str, Any]], tf: TF) -> Dict[int, Dict[str, float]]:
        """Limpia y valida los datos de una serie"""
        cleaned = {}
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        for ts, bar in series.items():
            try:
                # Validar timestamp
                if not isinstance(ts, int) or ts <= 0:
                    logger.warning(f"Timestamp inválido {ts} en TF {tf}")
                    continue
                
                # Validar y convertir campos OHLCV
                clean_bar = {}
                valid_bar = True
                
                for field in required_fields:
                    if field not in bar:
                        logger.warning(f"Campo {field} faltante en barra {ts} TF {tf}")
                        valid_bar = False
                        break
                    
                    try:
                        value = float(bar[field])
                        if field != 'volume' and value <= 0:
                            logger.warning(f"Precio inválido {field}={value} en barra {ts} TF {tf}")
                            valid_bar = False
                            break
                        if field == 'volume' and value < 0:
                            logger.warning(f"Volumen negativo {value} en barra {ts} TF {tf}")
                            valid_bar = False
                            break
                        clean_bar[field] = value
                    except (ValueError, TypeError):
                        logger.warning(f"No se puede convertir {field}={bar[field]} en barra {ts} TF {tf}")
                        valid_bar = False
                        break
                
                # Validar lógica OHLC
                if valid_bar:
                    o, h, l, c = clean_bar['open'], clean_bar['high'], clean_bar['low'], clean_bar['close']
                    if not (l <= o <= h and l <= c <= h and l <= h):
                        logger.warning(f"OHLC inválido en barra {ts} TF {tf}: O={o}, H={h}, L={l}, C={c}")
                        valid_bar = False
                
                if valid_bar:
                    cleaned[ts] = clean_bar
                    
            except Exception as e:
                logger.warning(f"Error procesando barra {ts} TF {tf}: {e}")
                
        return cleaned

    def _setup_timeline(self) -> None:
        """Configura el timeline base"""
        base_ts_list = self.ts_index_by_tf.get(self.base_tf, [])
        if not base_ts_list:
            raise NoDataError(f"No hay datos válidos para TF base {self.base_tf} en {self.symbol}/{self.market}/{self.stage}")
        
        # Aplicar filtros de rango
        if self.ts_from is not None:
            base_ts_list = [t for t in base_ts_list if t >= int(self.ts_from)]
        if self.ts_to is not None:
            base_ts_list = [t for t in base_ts_list if t <= int(self.ts_to)]
        
        if not base_ts_list:
            raise NoDataError(f"No hay datos en el rango especificado para TF base {self.base_tf}")
            
        self._base_ts_list = base_ts_list
        self._i = 0
        self._start_i = 0

    # ------------- API Pública -------------
    
    def now_ts(self) -> int:
        """Retorna el timestamp actual del timeline base"""
        if self._i >= len(self._base_ts_list):
            raise IndexError("Cursor fuera del rango de datos")
        return self._base_ts_list[self._i]

    def next(self) -> bool:
        """
        Avanza al siguiente timestamp en el timeline base.
        
        Returns:
            bool: True si se avanzó exitosamente, False si se alcanzó el final
        """
        if self._i < len(self._base_ts_list) - 1:
            self._i += 1
            return True
        return False
    
    def is_end_of_data(self) -> bool:
        """Verifica si se alcanzó el final de los datos"""
        return self._i >= len(self._base_ts_list) - 1
    
    def reset_to_start(self) -> None:
        """Reinicia el cursor al inicio del histórico"""
        self._i = self._start_i
        logger.debug(f"Cursor reiniciado a posición {self._start_i}")

    @property
    def range_ts(self) -> tuple[int, int]:
        """Retorna el rango de timestamps disponible (inicio, fin)"""
        if not self._base_ts_list:
            return (0, 0)
        return (self._base_ts_list[0], self._base_ts_list[-1])

    @property
    def current_position(self) -> tuple[int, int]:
        """Retorna la posición actual (índice actual, total)"""
        return (self._i, len(self._base_ts_list))

    @property
    def progress(self) -> float:
        """Retorna el progreso actual como porcentaje (0.0 a 1.0)"""
        if len(self._base_ts_list) <= 1:
            return 1.0
        return self._i / (len(self._base_ts_list) - 1)

    def get_price(self) -> Optional[float]:
        """Retorna el precio de cierre actual del TF base"""
        try:
            bar = self.get_bar(self.base_tf)
            return None if bar is None else bar.get("close")
        except Exception as e:
            logger.error(f"Error obteniendo precio: {e}")
            return None

    def get_bar(self, tf: TF) -> Optional[Dict[str, float]]:
        """
        Retorna la última barra con ts <= now_ts para el TF dado.
        
        Args:
            tf: Timeframe solicitado
            
        Returns:
            Diccionario con datos OHLCV o None si no hay datos
        """
        try:
            bars = self.series_by_tf.get(tf, {})
            ts_list = self.ts_index_by_tf.get(tf, [])
            
            if not ts_list:
                return None
                
            ts_now = self.now_ts()
            
            # Búsqueda binaria optimizada
            pos = bisect_right(ts_list, ts_now) - 1
            if pos < 0:
                return None
                
            ts = ts_list[pos]
            return bars.get(ts)
            
        except Exception as e:
            logger.error(f"Error obteniendo barra para TF {tf}: {e}")
            return None

    def get_bars_history(self, tf: TF, count: int = 1) -> List[Dict[str, float]]:
        """
        Retorna las últimas 'count' barras para el TF dado hasta now_ts.
        
        Args:
            tf: Timeframe solicitado
            count: Número de barras a retornar
            
        Returns:
            Lista de diccionarios con datos OHLCV (más reciente al final)
        """
        try:
            bars = self.series_by_tf.get(tf, {})
            ts_list = self.ts_index_by_tf.get(tf, [])
            
            if not ts_list or count <= 0:
                return []
                
            ts_now = self.now_ts()
            end_pos = bisect_right(ts_list, ts_now)
            start_pos = max(0, end_pos - count)
            
            result = []
            for i in range(start_pos, end_pos):
                if i < len(ts_list):
                    ts = ts_list[i]
                    bar = bars.get(ts)
                    if bar is not None:
                        result.append(bar.copy())
                        
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo historial para TF {tf}: {e}")
            return []

    def aligned_view(self, required_tfs: List[TF]) -> Dict[TF, Dict[str, float]]:
        """
        Retorna una vista alineada de las últimas barras para los TFs especificados.
        
        Args:
            required_tfs: Lista de timeframes requeridos
            
        Returns:
            Diccionario con TF como clave y datos OHLCV como valor
        """
        out: Dict[TF, Dict[str, float]] = {}
        
        for tf in required_tfs:
            try:
                bar = self.get_bar(tf)
                if bar is not None:
                    out[tf] = bar.copy()  # Copiar para evitar modificaciones accidentales
            except Exception as e:
                logger.error(f"Error en aligned_view para TF {tf}: {e}")
                
        return out

    def get_available_tfs(self) -> List[TF]:
        """Retorna la lista de timeframes disponibles con datos"""
        return [tf for tf in self.tfs if self.ts_index_by_tf.get(tf)]

    def get_data_info(self) -> Dict[str, Any]:
        """Retorna información sobre los datos cargados"""
        info = {
            'symbol': self.symbol,
            'market': self.market,
            'base_tf': self.base_tf,
            'stage': self.stage,
            'range_ts': self.range_ts,
            'total_bars_base': len(self._base_ts_list),
            'current_position': self.current_position,
            'progress': self.progress,
            'tfs_info': {}
        }
        
        for tf in self.tfs:
            ts_list = self.ts_index_by_tf.get(tf, [])
            info['tfs_info'][tf] = {
                'available': len(ts_list) > 0,
                'bar_count': len(ts_list),
                'range': (ts_list[0], ts_list[-1]) if ts_list else (0, 0)
            }
            
        return info