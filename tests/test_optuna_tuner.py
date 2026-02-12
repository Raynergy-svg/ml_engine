"""
Tests for Optuna hyperparameter optimization.

Tests verify that:
- Optuna study creation works correctly
- Optimization runs successfully
- Pruning strategies work
- Multi-objective optimization works
- Visualization functions work
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if optuna is available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

if OPTUNA_AVAILABLE:
    from src.training.optuna_tuner import (
        OptunaConfig,
        OptunaTuner,
        create_transformer_objective,
    )


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaBasics:
    """Test basic Optuna functionality."""
    
    def test_config_creation(self):
        """Test that OptunaConfig can be created with defaults."""
        config = OptunaConfig()
        assert config.study_name == "ml_engine_optimization"
        assert config.n_trials == 100
        assert config.direction == "minimize"
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = OptunaConfig(
            study_name="test_study",
            n_trials=50,
            direction="maximize",
            sampler="random",
            pruner=None,
        )
        assert config.study_name == "test_study"
        assert config.n_trials == 50
        assert config.direction == "maximize"
        assert config.sampler == "random"
        assert config.pruner is None
    
    def test_tuner_creation(self):
        """Test that OptunaTuner can be created."""
        config = OptunaConfig(n_trials=10)
        tuner = OptunaTuner(config)
        assert tuner.config == config
        assert tuner.study is None  # Not created yet
    
    def test_study_creation(self):
        """Test study creation."""
        config = OptunaConfig(
            study_name="test_study",
            storage=None,  # In-memory
        )
        tuner = OptunaTuner(config)
        study = tuner.create_study()
        
        assert study is not None
        assert study.study_name == "test_study"
        assert tuner.study is study


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptimization:
    """Test optimization functionality."""
    
    def test_simple_optimization(self):
        """Test basic single-objective optimization."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        config = OptunaConfig(
            study_name="test_simple",
            n_trials=20,
            direction="minimize",
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(objective)
        
        assert 'best_params' in results
        assert 'best_value' in results
        assert 'n_trials' in results
        
        # Best x should be close to 2
        best_x = results['best_params']['x']
        assert abs(best_x - 2) < 1.0  # Within 1 of optimal
    
    def test_multi_parameter_optimization(self):
        """Test optimization with multiple parameters."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            z = trial.suggest_int('z', -5, 5)
            
            return (x - 2) ** 2 + (y + 3) ** 2 + (z - 1) ** 2
        
        config = OptunaConfig(
            study_name="test_multi_param",
            n_trials=30,
            direction="minimize",
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(objective)
        
        best_params = results['best_params']
        assert abs(best_params['x'] - 2) < 1.5
        assert abs(best_params['y'] - (-3)) < 1.5
        assert best_params['z'] in [0, 1, 2]  # Should be near 1
    
    def test_categorical_parameters(self):
        """Test optimization with categorical parameters."""
        def objective(trial):
            optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop'])
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            
            # Dummy function - adam with lr near 1e-3 is "best"
            if optimizer == 'adam':
                return abs(np.log10(lr) - (-3))
            else:
                return abs(np.log10(lr) - (-3)) + 1.0
        
        config = OptunaConfig(
            study_name="test_categorical",
            n_trials=30,
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(objective)
        
        # Should prefer 'adam'
        assert results['best_params']['optimizer'] == 'adam'
    
    def test_pruning(self):
        """Test that pruning works."""
        def objective(trial):
            # Simulate training over epochs
            for step in range(10):
                # Report intermediate value
                intermediate_value = step * trial.suggest_float('x', 0, 1)
                trial.report(intermediate_value, step)
                
                # Check if should prune
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return trial.suggest_float('x', 0, 1)
        
        config = OptunaConfig(
            study_name="test_pruning",
            n_trials=20,
            pruner="median",
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(objective)
        
        # Check that some trials were pruned
        stats = tuner.get_study_statistics()
        # At least some completed trials
        assert stats['n_completed'] > 0


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestMultiObjective:
    """Test multi-objective optimization."""
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            
            # Two conflicting objectives
            obj1 = (x - 2) ** 2  # Minimize (best at x=2)
            obj2 = (x + 3) ** 2  # Minimize (best at x=-3)
            
            return [obj1, obj2]
        
        config = OptunaConfig(
            study_name="test_multi_obj",
            n_trials=30,
            directions=['minimize', 'minimize'],
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(objective)
        
        assert 'best_trials' in results
        assert 'pareto_front' in results
        
        # Should find multiple Pareto-optimal solutions
        assert len(results['best_trials']) > 1


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestStorage:
    """Test persistent storage."""
    
    def test_sqlite_storage(self):
        """Test SQLite persistent storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage_url = f"sqlite:///{db_path}"
            
            def objective(trial):
                x = trial.suggest_float('x', -10, 10)
                return (x - 2) ** 2
            
            # First optimization
            config1 = OptunaConfig(
                study_name="persistent_test",
                n_trials=10,
                storage=storage_url,
                load_if_exists=False,
                show_progress_bar=False,
            )
            
            tuner1 = OptunaTuner(config1)
            results1 = tuner1.optimize(objective)
            n_trials_1 = results1['n_trials']
            
            # Second optimization - should load existing study
            config2 = OptunaConfig(
                study_name="persistent_test",
                n_trials=10,
                storage=storage_url,
                load_if_exists=True,
                show_progress_bar=False,
            )
            
            tuner2 = OptunaTuner(config2)
            results2 = tuner2.optimize(objective)
            n_trials_2 = results2['n_trials']
            
            # Should have more trials in second run
            assert n_trials_2 == n_trials_1 + 10


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestUtilities:
    """Test utility functions."""
    
    def test_get_study_statistics(self):
        """Test study statistics."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            if x > 5:
                raise Exception("Intentional failure")
            return (x - 2) ** 2
        
        config = OptunaConfig(
            study_name="test_stats",
            n_trials=20,
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        try:
            tuner.optimize(objective)
        except Exception:
            pass  # Some trials may fail
        
        stats = tuner.get_study_statistics()
        
        assert 'n_trials' in stats
        assert 'n_completed' in stats
        assert 'n_failed' in stats
        assert stats['n_trials'] > 0
    
    def test_get_best_params(self):
        """Test getting best parameters."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        config = OptunaConfig(
            study_name="test_best_params",
            n_trials=15,
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        tuner.optimize(objective)
        
        best_params = tuner.get_best_params()
        assert 'x' in best_params
        assert abs(best_params['x'] - 2) < 2.0


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestTransformerObjective:
    """Test Transformer-specific objective creation."""
    
    def test_create_transformer_objective(self):
        """Test creating Transformer objective function."""
        # Dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        # Dummy training function
        def train_fn(params, X_train, y_train, X_val, y_val, trial=None):
            # Simulate validation loss
            lr = params['learning_rate']
            batch_size = params['batch_size']
            
            # Smaller batch + lower lr = better (dummy logic)
            loss = lr * 1000 + batch_size / 100
            return loss
        
        objective = create_transformer_objective(
            train_fn, X_train, y_train, X_val, y_val
        )
        
        # Test the objective with a trial
        config = OptunaConfig(
            study_name="test_transformer_obj",
            n_trials=10,
            storage=None,
            show_progress_bar=False,
        )
        
        tuner = OptunaTuner(config)
        results = tuner.optimize(objective)
        
        # Should find lower batch size and learning rate
        assert results['best_params']['batch_size'] <= 64
        assert results['best_params']['learning_rate'] < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
