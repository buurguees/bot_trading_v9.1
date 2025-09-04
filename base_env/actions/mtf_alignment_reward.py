# base_env/actions/mtf_alignment_reward.py
"""
Sistema de rewards por alineación multi-timeframe (confluencia).
Bonus si el trade va a favor de la dirección de timeframes superiores.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


class MTFAlignmentReward:
    """Sistema de rewards por alineación multi-timeframe"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de rewards por alineación MTF
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        mtf_config = self.config.get("mtf_alignment", {})
        self.enabled = mtf_config.get("enabled", True)
        self.higher_tfs = mtf_config.get("higher_tf", ["1h", "4h"])
        self.agree_bonus = mtf_config.get("agree_bonus", 0.15)
        self.disagree_penalty = mtf_config.get("disagree_penalty", 0.075)

    def calculate_mtf_alignment_reward(self, trade_side: int, obs: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward por alineación con timeframes superiores
        
        Args:
            trade_side: Dirección del trade (+1 long, -1 short)
            obs: Observación del entorno con datos de timeframes
            
        Returns:
            Tupla (reward, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if not self.enabled or trade_side == 0:
            return total_reward, reward_components
        
        # Obtener direcciones de timeframes superiores
        tf_directions = []
        for tf in self.higher_tfs:
            tf_direction = self._get_tf_direction(obs, tf)
            if tf_direction is not None:
                tf_directions.append(tf_direction)
        
        if not tf_directions:
            return total_reward, reward_components
        
        # Calcular alineación promedio
        alignment_score = 0.0
        for tf_dir in tf_directions:
            if tf_dir * trade_side > 0:  # Misma dirección
                alignment_score += 1.0
            elif tf_dir * trade_side < 0:  # Dirección opuesta
                alignment_score -= 1.0
            # Si tf_dir == 0 (neutral), no contribuye
        
        # Normalizar por número de timeframes
        avg_alignment = alignment_score / len(tf_directions)
        
        # Aplicar reward/penalty
        if avg_alignment > 0.5:  # Mayoría a favor
            total_reward = self.agree_bonus
            reward_components["mtf_alignment_bonus"] = total_reward
        elif avg_alignment < -0.5:  # Mayoría en contra
            total_reward = -self.disagree_penalty
            reward_components["mtf_alignment_penalty"] = total_reward
        
        reward_components["avg_alignment"] = avg_alignment
        reward_components["tf_directions"] = tf_directions
        
        return total_reward, reward_components

    def _get_tf_direction(self, obs: Dict[str, Any], tf: str) -> Optional[int]:
        """
        Extrae la dirección del timeframe desde la observación
        
        Args:
            obs: Observación del entorno
            tf: Timeframe a analizar
            
        Returns:
            Dirección (+1, -1, 0) o None si no se puede determinar
        """
        try:
            # Intentar obtener desde analysis.hierarchical
            analysis = obs.get("analysis", {})
            hierarchical = analysis.get("hierarchical", {})
            
            # Buscar en confirm_tfs o trend_tfs
            for tf_key in ["confirm_tfs", "trend_tfs"]:
                tf_data = hierarchical.get(tf_key, {})
                if tf in tf_data:
                    side_hint = tf_data[tf].get("side_hint", 0)
                    if side_hint != 0:
                        return int(side_hint)
            
            # Fallback: usar features técnicos simples
            features = obs.get("features", {})
            tf_features = features.get(tf, {})
            
            # Usar EMA crossover como proxy de dirección
            ema_fast = tf_features.get("ema12", 0)
            ema_slow = tf_features.get("ema26", 0)
            
            if ema_fast > 0 and ema_slow > 0:
                if ema_fast > ema_slow:
                    return 1  # Bullish
                elif ema_fast < ema_slow:
                    return -1  # Bearish
            
            return 0  # Neutral
            
        except (KeyError, ValueError, TypeError):
            return None

    def reset(self):
        """Resetea el sistema"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "enabled": self.enabled,
            "higher_tfs": self.higher_tfs,
            "agree_bonus": self.agree_bonus,
            "disagree_penalty": self.disagree_penalty
        }
