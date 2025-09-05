"""
Tests para verificar que no hay valores hardcodeados que deberían venir de YAML.
"""
import pytest
import re
from pathlib import Path


def test_no_hardcoded_values():
    """Test que no hay valores hardcodeados en el código."""
    # Patrones de valores que deberían venir de YAML
    hardcoded_patterns = [
        # Valores de riesgo
        (r'risk_pct_per_trade\s*=\s*[0-9.]+', "risk_pct_per_trade debería venir de risk.yaml"),
        (r'min_sl_pct\s*=\s*[0-9.]+', "min_sl_pct debería venir de risk.yaml"),
        (r'leverage\s*=\s*[0-9.]+', "leverage debería venir de symbols.yaml"),
        
        # Valores de bankruptcy
        (r'threshold_pct\s*=\s*[0-9.]+', "threshold_pct debería venir de risk.yaml"),
        (r'penalty_reward\s*=\s*-?[0-9.]+', "penalty_reward debería venir de risk.yaml"),
        
        # Valores de fees
        (r'taker_fee_bps\s*=\s*[0-9.]+', "taker_fee_bps debería venir de fees.yaml"),
        (r'maker_fee_bps\s*=\s*[0-9.]+', "maker_fee_bps debería venir de fees.yaml"),
        
        # Valores de rewards
        (r'w_[a-z_]+\s*=\s*[0-9.-]+', "pesos de rewards deberían venir de rewards.yaml"),
        
        # Valores de configuración
        (r'min_confidence\s*=\s*[0-9.]+', "min_confidence debería venir de hierarchical.yaml"),
        (r'warmup_bars\s*=\s*[0-9]+', "warmup_bars debería venir de train.yaml"),
        (r'n_steps\s*=\s*[0-9]+', "n_steps debería venir de train.yaml"),
        
        # Valores de balance
        (r'initial_balance\s*=\s*[0-9.]+', "initial_balance debería venir de train.yaml"),
        (r'target_balance\s*=\s*[0-9.]+', "target_balance debería venir de train.yaml"),
    ]
    
    # Archivos a verificar
    code_files = [
        "base_env/base_env.py",
        "base_env/risk/manager.py",
        "base_env/accounting/ledger.py",
        "train_env/gym_wrapper.py",
        "train_env/reward_shaper.py",
        "train_env/vec_factory.py",
        "scripts/train_ppo.py"
    ]
    
    violations = []
    
    for file_path in code_files:
        if not Path(file_path).exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for pattern, message in hardcoded_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Excluir comentarios y docstrings
                if not _is_in_comment_or_docstring(content, match):
                    violations.append(f"{file_path}: {match} - {message}")
    
    # Verificar que no hay violaciones
    if violations:
        print("🚨 VALORES HARDCODEADOS DETECTADOS:")
        for violation in violations:
            print(f"   {violation}")
        assert False, f"Se encontraron {len(violations)} valores hardcodeados que deberían venir de YAML"


def _is_in_comment_or_docstring(content: str, match: str) -> bool:
    """Verifica si un match está en un comentario o docstring."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if match in line:
            # Verificar si está en comentario
            if line.strip().startswith('#') or '#' in line and line.find('#') < line.find(match):
                return True
            
            # Verificar si está en docstring
            if '"""' in line or "'''" in line:
                return True
    
    return False


def test_config_files_exist():
    """Test que todos los archivos de configuración existen."""
    required_configs = [
        "config/symbols.yaml",
        "config/risk.yaml",
        "config/fees.yaml",
        "config/hierarchical.yaml",
        "config/pipeline.yaml",
        "config/train.yaml",
        "config/rewards.yaml"
    ]
    
    missing_configs = []
    for config_file in required_configs:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
    
    assert not missing_configs, f"Archivos de configuración faltantes: {missing_configs}"


def test_config_loader_works():
    """Test que el config loader funciona correctamente."""
    from base_env.config.config_loader import config_loader
    
    # Test que puede cargar todos los archivos
    symbols = config_loader.load_symbols()
    assert len(symbols) > 0, "No se pudieron cargar símbolos"
    
    risk_config = config_loader.load_risk_config()
    assert risk_config is not None, "No se pudo cargar risk config"
    
    fees_config = config_loader.load_fees_config()
    assert fees_config is not None, "No se pudo cargar fees config"
    
    hierarchical_config = config_loader.load_hierarchical_config()
    assert hierarchical_config is not None, "No se pudo cargar hierarchical config"
    
    pipeline_config = config_loader.load_pipeline_config()
    assert pipeline_config is not None, "No se pudo cargar pipeline config"
    
    train_config = config_loader.load_train_config()
    assert train_config is not None, "No se pudo cargar train config"
    
    rewards_config = config_loader.load_rewards_config()
    assert rewards_config is not None, "No se pudo cargar rewards config"


def test_env_config_creation():
    """Test que se puede crear una configuración completa del entorno."""
    from base_env.config.config_loader import config_loader
    
    # Test con BTCUSDT
    try:
        env_config = config_loader.create_env_config("BTCUSDT", "train_spot")
        assert env_config.symbol_meta.symbol == "BTCUSDT"
        assert env_config.market == "spot"
        assert env_config.mode == "train_spot"
    except ValueError:
        # Si BTCUSDT no está configurado, usar el primer símbolo disponible
        symbols = config_loader.load_symbols()
        if symbols:
            symbol = symbols[0].symbol
            env_config = config_loader.create_env_config(symbol, "train_spot")
            assert env_config.symbol_meta.symbol == symbol


def test_config_validation():
    """Test que las configuraciones tienen valores válidos."""
    from base_env.config.config_loader import config_loader
    
    # Test risk config
    risk_config = config_loader.load_risk_config()
    assert 0 < risk_config.common.bankruptcy.threshold_pct <= 100, "threshold_pct debe estar entre 0 y 100"
    assert risk_config.common.bankruptcy.mode in ["end", "soft_reset"], "modo de bankruptcy inválido"
    assert risk_config.spot.risk_pct_per_trade > 0, "risk_pct_per_trade debe ser positivo"
    assert risk_config.futures.risk_pct_per_trade > 0, "risk_pct_per_trade debe ser positivo"
    
    # Test symbols config
    symbols = config_loader.load_symbols()
    for symbol in symbols:
        assert symbol.symbol, "Símbolo debe tener nombre"
        assert symbol.market in ["spot", "futures"], "Market debe ser spot o futures"
        assert len(symbol.enabled_tfs) > 0, "Debe tener al menos un TF habilitado"
        if symbol.leverage:
            assert symbol.leverage.min > 0, "Leverage min debe ser positivo"
            assert symbol.leverage.max >= symbol.leverage.min, "Leverage max debe ser >= min"
    
    # Test hierarchical config
    hier_config = config_loader.load_hierarchical_config()
    assert 0 <= hier_config.min_confidence <= 1, "min_confidence debe estar entre 0 y 1"
    assert len(hier_config.execute_tfs) > 0, "Debe tener al menos un TF de ejecución"


if __name__ == "__main__":
    pytest.main([__file__])
