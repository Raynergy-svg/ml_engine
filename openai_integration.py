import os
import json
import time
import logging
import yaml

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if OPENAI_AVAILABLE and openai:
    openai.api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")


def set_openai_credentials(config: dict) -> None:
    """Set OpenAI credentials from config or environment variables."""
    if not OPENAI_AVAILABLE:
        logging.warning("OpenAI not installed, skipping credential setup")
        return
    openai.api_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.error("OpenAI API key not found in configuration or .env")
        raise ValueError("Missing OpenAI API key")


client = openai


def query_openai(prompt, model="gpt-4", temperature=0.2, max_attempts=3):
    """General-purpose function to query the OpenAI API."""
    if not OPENAI_AVAILABLE:
        logging.error("OpenAI not installed.")
        return None
    if not openai.api_key:
        logging.error("OPENAI_API_KEY not set.")
        return None
    attempts = 0
    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a production-level ML optimizer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            attempts += 1
            logging.error(f"OpenAI query error (attempt {attempts}): {e}")
            time.sleep(1)
    return None


def query_for_learning_rate(metrics, model="gpt-4", temperature=0.2):
    """Query OpenAI to suggest a new learning rate."""
    prompt = (
        "You are a production-level ML optimizer. "
        "I am training a model on a CPU-only machine with 6 cores "
        "and 16GB RAM.\n"
        f"Here are the current training metrics in JSON format:\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        "Based on these metrics, please suggest a new learning rate "
        "in valid JSON format, for example: "
        '{"learning_rate": 0.004}. Return only valid JSON.'
    )
    response_text = query_openai(prompt, model=model, temperature=temperature)
    if response_text is None:
        return None
    try:
        suggestion = json.loads(response_text)
        return suggestion.get("learning_rate")
    except Exception as e:
        logging.error(f"Error parsing learning rate response: {e}")
        return None


def query_for_explanation(explanation_features, model="gpt-4", temperature=0.2):
    """Query OpenAI to generate a concise textual explanation."""
    prompt = (
        "You are a financial analyst. Based on the following "
        "explanation features extracted from a model's output, "
        "generate a concise explanation of why the stock might be "
        "moving up or down.\n"
        f"Explanation features: {json.dumps(explanation_features, indent=2)}\n"
        "Return only a plain text explanation."
    )
    response_text = query_openai(prompt, model=model, temperature=temperature)
    return response_text


def query_for_loss_function_adjustment(
    metrics, current_loss_functions, model="gpt-4", temperature=0.2
):
    """Query OpenAI for suggestions on adjusting loss functions."""
    prompt = (
        "You are a production-level ML optimizer. "
        "I am training a multi-task model for financial forecasting "
        "and risk management.\n"
        f"Metrics: {json.dumps(metrics, indent=2)}\n"
        f"Current Loss Functions: {json.dumps(current_loss_functions, indent=2)}\n\n"
        "Based on these metrics, please suggest adjustments or "
        "alternative loss functions in valid JSON format, "
        'for example: {"price_loss": "MSELoss", "explanation_loss": "L1Loss", "risk_loss": "HuberLoss"}. '
        "Return only valid JSON."
    )
    response_text = query_openai(prompt, model=model, temperature=temperature)
    if response_text is None:
        return None
    try:
        suggestions = json.loads(response_text)
        return suggestions
    except Exception as e:
        logging.error(f"Error parsing loss function suggestion response: {e}")
        return None


def query_for_data_handling_adjustment(
    metrics, data_characteristics, model="gpt-4", temperature=0.2
):
    """Query OpenAI for recommendations on improving data handling."""
    prompt = (
        "You are a production-level ML optimizer. "
        "I am training a multi-task model for financial forecasting "
        "and risk management.\n"
        f"Metrics: {json.dumps(metrics, indent=2)}\n"
        f"Data Characteristics: {json.dumps(data_characteristics, indent=2)}\n\n"
        "Based on this information, please suggest adjustments for "
        "data handling in valid JSON format, such as changes in batch "
        "size, data augmentation, or preprocessing steps. "
        "Return only valid JSON."
    )
    response_text = query_openai(prompt, model=model, temperature=temperature)
    if response_text is None:
        return None
    try:
        suggestions = json.loads(response_text)
        return suggestions
    except Exception as e:
        logging.error(f"Error parsing data handling suggestion response: {e}")
        return None


def query_for_model_architecture_adjustment(
    metrics, current_architecture, model="gpt-4", temperature=0.2
):
    """Query OpenAI for suggestions on adjusting the model architecture."""
    prompt = (
        "You are a production-level ML optimizer. "
        "I am training a multi-task model for financial forecasting "
        "and risk management.\n"
        f"Metrics: {json.dumps(metrics, indent=2)}\n"
        f"Current Architecture: {json.dumps(current_architecture, indent=2)}\n\n"
        "Based on this information, please suggest adjustments for "
        "the model architecture in valid JSON format, "
        'for example: {"hidden_size": 256, "num_layers": 4, "dropout": 0.3}. '
        "Return only valid JSON."
    )
    response_text = query_openai(prompt, model=model, temperature=temperature)
    if response_text is None:
        return None
    try:
        suggestions = json.loads(response_text)
        return suggestions
    except Exception as e:
        logging.error(f"Error parsing model architecture suggestion response: {e}")
        return None


def query_for_auto_configuration(metrics, current_config):
    """Query GPT for configuration updates."""
    prompt = (
        "Given the current training metrics:\n"
        f"{metrics}\n"
        "and the current configuration:\n"
        f"{current_config}\n"
        "Provide ONLY valid YAML configuration changes to improve "
        "model performance. "
        "Ensure the response is ONLY YAML and nothing else "
        "(no surrounding text or code blocks)."
    )
    response = query_openai(prompt)
    try:
        if response:
            response = response.replace("```yaml", "").replace("```", "").strip()
        config_updates = yaml.safe_load(response)
    except Exception as e:
        logging.error("Failed to parse GPT response: %s\nResponse was: %s", e, response)
        config_updates = {}

    logging.debug("Config Updates:\n%s", config_updates)
    return config_updates


def report_config_changes(current_config, config_updates, config_path: str = "config_tuned.yaml"):
    """Merge config with updates and write to the given config_path."""
    import yaml

    updated_config = current_config.copy()
    for key, value in config_updates.items():
        if key not in updated_config:
            logging.info(f"[OpenAI Update] New config key '{key}' added: {value}")
        else:
            old_val = updated_config[key]
            if old_val != value:
                logging.info(
                    f"[OpenAI Update] '{key}' changed from {old_val} to {value}"
                )
        updated_config[key] = value

    logging.debug(f"Updated Config before writing to file:\n{updated_config}")

    with open(str(config_path), "w") as f:
        yaml.dump(updated_config, f)
    return updated_config


def autotune_configurations(config: dict) -> dict:
    """Use optuna to autotune ML engine configurations (legacy simple version)."""
    if not OPTUNA_AVAILABLE:
        logging.warning("Optuna not installed, skipping autotune")
        return config

    def objective(trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        score = (lr - 0.001) ** 2
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    tuned_params = study.best_params
    if "model" not in config:
        config["model"] = {}
    config["model"].update(tuned_params)
    logging.info(f"Autotuned configuration: {tuned_params}")
    return config


# =============================================================================
# FULL ENSEMBLE OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

# SQLite storage for cross-session persistence
OPTUNA_DB_PATH = "trained_data/optuna.db"


def get_optuna_storage() -> str:
    """Get SQLite storage URL for Optuna."""
    from pathlib import Path
    Path("trained_data").mkdir(exist_ok=True)
    return f"sqlite:///{OPTUNA_DB_PATH}"


def autotune_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    n_trials: int = 100,
    study_name: str = "ensemble_hpo",
    timeout_hours: float = 4.0,
    models_to_tune: list = None,
) -> dict:
    """
    Full ensemble hyperparameter optimization using Optuna with SQLite persistence.
    
    Tunes all 5 models in the ensemble:
    - Transformer: learning_rate, dropout, d_model, num_heads, num_layers
    - XGBoost: n_estimators, max_depth, learning_rate
    - Random Forest: n_estimators, max_depth, min_samples_split
    - Ridge: alpha
    - HistGradientBoosting: max_iter, max_depth, learning_rate
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials (default: 100)
        study_name: Name for the Optuna study (persisted in SQLite)
        timeout_hours: Maximum time for optimization
        models_to_tune: List of models to tune (default: all)
        
    Returns:
        Dict with best hyperparameters for each model
    """
    if not OPTUNA_AVAILABLE:
        logging.error("Optuna not installed. Install with: pip install optuna")
        return {}
    
    import numpy as np
    from sklearn.metrics import accuracy_score, r2_score
    
    models_to_tune = models_to_tune or ["transformer", "xgboost", "rf", "ridge", "histgb"]
    storage = get_optuna_storage()
    
    best_params = {}
    
    # === TRANSFORMER HYPERPARAMETERS ===
    if "transformer" in models_to_tune:
        logging.info("🔧 Tuning Transformer hyperparameters...")
        
        def transformer_objective(trial: optuna.trial.Trial) -> float:
            # Hyperparameter search space
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
                "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dff": trial.suggest_categorical("dff", [32, 64, 128, 256]),
            }
            
            # Ensure d_model is divisible by num_heads
            if params["d_model"] % params["num_heads"] != 0:
                return 0.0  # Invalid config
            
            try:
                # Quick training with reduced epochs for HPO
                from modular_trainers import TransformerTrainer, TrainerConfig
                
                config = TrainerConfig(
                    epochs=30,  # Reduced for HPO
                    batch_size=128,
                    learning_rate=params["learning_rate"],
                    patience=5,
                    transformer_d_model=params["d_model"],
                    transformer_num_heads=params["num_heads"],
                    transformer_num_layers=params["num_layers"],
                    transformer_dff=params["dff"],
                    transformer_dropout=params["dropout"],
                )
                
                trainer = TransformerTrainer(
                    X_train, y_train, X_val, y_val,
                    config=config,
                )
                trainer.train()
                
                # Evaluate
                preds = trainer.predict(X_val)
                if hasattr(preds, 'argmax'):
                    pred_classes = preds.argmax(axis=1)
                else:
                    pred_classes = (np.array(preds) > 0.5).astype(int)
                
                accuracy = accuracy_score(y_val, pred_classes)
                return accuracy
                
            except Exception as e:
                logging.warning(f"Trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(
            study_name=f"{study_name}_transformer",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            transformer_objective,
            n_trials=n_trials // 3,  # Allocate trials across models
            timeout=timeout_hours * 3600 / 3,
            show_progress_bar=True,
        )
        best_params["transformer"] = study.best_params
        logging.info(f"✅ Transformer best: {study.best_params} (acc={study.best_value:.4f})")
    
    # === XGBOOST HYPERPARAMETERS ===
    if "xgboost" in models_to_tune:
        logging.info("🔧 Tuning XGBoost hyperparameters...")
        
        def xgb_objective(trial: optuna.trial.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            
            try:
                from xgboost import XGBClassifier
                
                model = XGBClassifier(
                    **params,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val, preds)
                return accuracy
                
            except Exception as e:
                logging.warning(f"XGB trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(
            study_name=f"{study_name}_xgboost",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            xgb_objective,
            n_trials=n_trials // 5,
            timeout=timeout_hours * 3600 / 5,
            show_progress_bar=True,
        )
        best_params["xgboost"] = study.best_params
        logging.info(f"✅ XGBoost best: {study.best_params} (acc={study.best_value:.4f})")
    
    # === RANDOM FOREST HYPERPARAMETERS ===
    if "rf" in models_to_tune:
        logging.info("🔧 Tuning Random Forest hyperparameters...")
        
        def rf_objective(trial: optuna.trial.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }
            
            try:
                from sklearn.ensemble import RandomForestClassifier
                
                model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val, preds)
                return accuracy
                
            except Exception as e:
                logging.warning(f"RF trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(
            study_name=f"{study_name}_rf",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            rf_objective,
            n_trials=n_trials // 5,
            timeout=timeout_hours * 3600 / 5,
            show_progress_bar=True,
        )
        best_params["rf"] = study.best_params
        logging.info(f"✅ RF best: {study.best_params} (acc={study.best_value:.4f})")
    
    # === RIDGE HYPERPARAMETERS ===
    if "ridge" in models_to_tune:
        logging.info("🔧 Tuning Ridge hyperparameters...")
        
        def ridge_objective(trial: optuna.trial.Trial) -> float:
            alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            
            try:
                from sklearn.linear_model import Ridge
                
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                r2 = r2_score(y_val, preds)
                return r2
                
            except Exception as e:
                logging.warning(f"Ridge trial failed: {e}")
                return -1.0
        
        study = optuna.create_study(
            study_name=f"{study_name}_ridge",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            ridge_objective,
            n_trials=n_trials // 10,  # Ridge is fast
            timeout=timeout_hours * 3600 / 10,
            show_progress_bar=True,
        )
        best_params["ridge"] = study.best_params
        logging.info(f"✅ Ridge best: {study.best_params} (r2={study.best_value:.4f})")
    
    # === HISTGRADIENTBOOSTING HYPERPARAMETERS ===
    if "histgb" in models_to_tune:
        logging.info("🔧 Tuning HistGradientBoosting hyperparameters...")
        
        def histgb_objective(trial: optuna.trial.Trial) -> float:
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            }
            
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier
                
                model = HistGradientBoostingClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val, preds)
                return accuracy
                
            except Exception as e:
                logging.warning(f"HistGB trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(
            study_name=f"{study_name}_histgb",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            histgb_objective,
            n_trials=n_trials // 5,
            timeout=timeout_hours * 3600 / 5,
            show_progress_bar=True,
        )
        best_params["histgb"] = study.best_params
        logging.info(f"✅ HistGB best: {study.best_params} (acc={study.best_value:.4f})")
    
    # Save best params to file
    import json
    from pathlib import Path
    
    params_path = Path("trained_data/optuna_best_params.json")
    params_path.write_text(json.dumps(best_params, indent=2))
    logging.info(f"📊 Best hyperparameters saved to {params_path}")
    
    return best_params


def load_best_params() -> dict:
    """Load best hyperparameters from previous Optuna runs."""
    import json
    from pathlib import Path
    
    params_path = Path("trained_data/optuna_best_params.json")
    if params_path.exists():
        return json.loads(params_path.read_text())
    return {}


def get_study_stats(study_name: str = "ensemble_hpo") -> dict:
    """Get statistics from Optuna studies."""
    storage = get_optuna_storage()
    stats = {}
    
    for model in ["transformer", "xgboost", "rf", "ridge", "histgb"]:
        try:
            study = optuna.load_study(
                study_name=f"{study_name}_{model}",
                storage=storage,
            )
            stats[model] = {
                "n_trials": len(study.trials),
                "best_value": study.best_value,
                "best_params": study.best_params,
            }
        except Exception:
            stats[model] = {"n_trials": 0, "best_value": None, "best_params": None}
    
    return stats
# — Raynergy-svg —
