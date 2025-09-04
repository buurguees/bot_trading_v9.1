# base_env/actions/bankruptcy_manager.py
"""
Gestor de bancarrota configurable.
Permite modo "end" (terminar episodio) o "soft_reset" (reiniciar balance y continuar)
según `config/risk.yaml`.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple


class BankruptcyManager:
	"""Encapsula la lógica de bancarrota y soft reset."""

	def __init__(self, env_ref):
		# Mantiene una referencia débil al entorno para operar sobre su estado
		self.env = env_ref

	def handle_bankruptcy(self, reward: float, ts_now: int, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
		"""
		Gestiona la bancarrota según configuración (end | soft_reset).
		Devuelve (obs, reward, done, info)
		"""
		env = self.env
		env._bankruptcy_detected = True
		penalty_reward = float(env.cfg.risk.common.bankruptcy.penalty_reward)
		reward += penalty_reward
		mode = env.cfg.risk.common.bankruptcy.mode

		# Cerrar posición si existe
		if env.pos.side != 0:
			env._close_position_force()

		if mode == "end":
			# Terminar episodio
			env._run_logger.finish(
				final_balance=env.portfolio.cash_quote,
				final_equity=env.portfolio.equity_quote,
				ts_end=ts_now,
				bankruptcy=True,
				penalty_reward=penalty_reward
			)
			return obs, reward, True, {"done_reason": "BANKRUPTCY", "bankruptcy": True, "penalty_reward": penalty_reward}

		if mode == "soft_reset":
			return self._soft_reset(penalty_reward, ts_now, obs)

		# Fallback a END
		env._run_logger.finish(
			final_balance=env.portfolio.cash_quote,
			final_equity=env.portfolio.equity_quote,
			ts_end=ts_now,
			bankruptcy=True,
			penalty_reward=penalty_reward
		)
		return obs, reward, True, {"done_reason": "BANKRUPTCY", "bankruptcy": True, "penalty_reward": penalty_reward}

	def _soft_reset(self, penalty_reward: float, ts_now: int, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
		"""Aplica soft reset: reinicia balance y continúa el episodio con cooldown/cap si procede."""
		env = self.env
		soft_reset_cfg = env.cfg.risk.common.bankruptcy.soft_reset

		# Máximo de resets
		if env._soft_reset_count >= soft_reset_cfg.max_resets_per_run:
			# Si excede, terminar episodio
			env._run_logger.finish(
				final_balance=env.portfolio.cash_quote,
				final_equity=env.portfolio.equity_quote,
				ts_end=ts_now,
				bankruptcy=True,
				penalty_reward=penalty_reward
			)
			return obs, penalty_reward, True, {"done_reason": "BANKRUPTCY_MAX_RESETS", "bankruptcy": True, "penalty_reward": penalty_reward}

		# Contabilidad del reset
		env._soft_reset_count += 1
		env._current_segment_id += 1

		# Reiniciar balance/equity y margen
		env.portfolio.cash_quote = float(env._init_cash)
		env.portfolio.equity_quote = float(env._init_cash)
		env.portfolio.used_margin = 0.0

		# Cooldown y leverage cap
		env._cooldown_bars_remaining = soft_reset_cfg.cooldown_bars
		env._leverage_cap_active = soft_reset_cfg.post_reset_leverage_cap

		# Logger
		env._run_logger.update_segment_id(env._current_segment_id)
		env._run_logger.update_soft_reset_count(env._soft_reset_count)

		return obs, penalty_reward, False, {
			"done_reason": "SOFT_RESET",
			"soft_reset": True,
			"reset_count": env._soft_reset_count,
			"segment_id": env._current_segment_id,
			"cooldown_bars": env._cooldown_bars_remaining,
			"leverage_cap": env._leverage_cap_active
		}
