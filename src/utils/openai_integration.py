"""
================================================================================
DEPRECATION NOTICE - openai_integration.py
================================================================================
Status: PARTIALLY ACTIVE - Used by main.py for `openai-tune` command

This module is still used by:
- main.py (lines 6333-6375): `openai-tune` CLI command
- tests/test_quant_critic.py: Unit tests for quant critic functionality

However, llm_providers.py is the preferred module for LLM operations:
- Supports multiple backends (Claude, OpenAI, Ollama)
- Has automatic provider selection based on available API keys
- More modern architecture with retry logic and better error handling

MIGRATION PATH:
- For new LLM integrations, use llm_providers.py instead
- The query_quant_critic() and improve_trading_rationale() functions are unique
  to this module and should be migrated to llm_providers.py if still needed
- autotune_configurations() uses Optuna and could be moved to a dedicated
  hyperparameter tuning module

UNIQUE FUNCTIONS TO PRESERVE:
- query_quant_critic(): Provides trading rationale critique (active, has tests)
- improve_trading_rationale(): Convenience wrapper for quant critic
- autotune_configurations(): Optuna-based hyperparameter tuning
- query_for_auto_configuration(): GPT-based config suggestions

DO NOT DELETE: Still has active usages in main.py
================================================================================
"""

import os
import json
import time
import logging
import openai
import optuna
import yaml

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")
logger = logging.getLogger(__name__)


def set_openai_credentials(config: dict) -> None:
    """Set OpenAI credentials from config or environment variables."""
    openai.api_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key not found in configuration or .env")
        raise ValueError("Missing OpenAI API key")


client = openai


def query_openai(prompt, model="gpt-4", temperature=0.2, max_attempts=3):
    """General-purpose function to query the OpenAI API."""
    if not openai.api_key:
        logger.error("OPENAI_API_KEY not set.")
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
            logger.error(f"OpenAI query error (attempt {attempts}): {e}")
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
        logger.error(f"Error parsing learning rate response: {e}")
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
        logger.error(f"Error parsing loss function suggestion response: {e}")
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
        logger.error(f"Error parsing data handling suggestion response: {e}")
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
        logger.error(f"Error parsing model architecture suggestion response: {e}")
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
        logger.error("Failed to parse GPT response: %s\nResponse was: %s", e, response)
        config_updates = {}

    logger.debug("Config Updates:\n%s", config_updates)
    return config_updates


def report_config_changes(current_config, config_updates, config_path: str = "config_tuned.yaml"):
    """Merge config with updates and write to the given config_path."""
    import yaml

    updated_config = current_config.copy()
    for key, value in config_updates.items():
        if key not in updated_config:
            logger.info(f"[OpenAI Update] New config key '{key}' added: {value}")
        else:
            old_val = updated_config[key]
            if old_val != value:
                logger.info(
                    f"[OpenAI Update] '{key}' changed from {old_val} to {value}"
                )
        updated_config[key] = value

    logger.debug(f"Updated Config before writing to file:\n{updated_config}")

    with open(str(config_path), "w") as f:
        yaml.dump(updated_config, f)
    return updated_config


def autotune_configurations(config: dict) -> dict:
    """Use optuna to autotune ML engine configurations."""

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
    logger.info(f"Autotuned configuration: {tuned_params}")
    return config


def query_quant_critic(
    ticker: str,
    timeframe: str,
    buddy_raw: dict,
    current_response: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> dict:
    """Query OpenAI to act as a quant critic improving trading rationale.

    This function provides specific feedback on Buddy's ML model predictions,
    identifying errors in logic, missing risks, overconfidence, or poor
    linkage to features.

    Args:
        ticker: The trading instrument (e.g., "USD_JPY", "EUR_USD")
        timeframe: The timeframe of the prediction (e.g., "H1", "M5")
        buddy_raw: Dict containing Buddy's raw output (confidence score, prob, etc.)
        current_response: The current interpretation of Buddy's prediction
        model: OpenAI model to use (default: "gpt-4")
        temperature: Temperature for generation (default: 0.2)

    Returns:
        Dict with keys:
        - feedback: str - Specific feedback on the interpretation (always present)
        - improved_rationale: str | None - Improved rationale, or None if optimal
        - risk_assessment: str | None - Additional risk factors, or None if not applicable
        - is_optimal: bool - True if no improvements needed, False otherwise
    """
    prompt = f"""You are an expert quant critic improving trading rationale from a static ML model (Buddy).

Original Task: Interpret Buddy's prediction for {ticker} on {timeframe}.
Buddy Raw Output: {json.dumps(buddy_raw, indent=2)}

Current Interpretation:
{current_response}

Provide specific feedback:
- Errors in logic, missing risks, overconfidence, poor linkage to features.
- Improvements for clarity, completeness, risk assessment.
- If already optimal: Say "No improvements needed."

Return your response in the following JSON format:
{{
    "feedback": "Specific feedback on the current interpretation",
    "improved_rationale": "Improved version of the rationale, or null if optimal",
    "risk_assessment": "Additional risk factors to consider",
    "is_optimal": true/false
}}

Return only valid JSON."""

    response_text = query_openai(prompt, model=model, temperature=temperature)
    if response_text is None:
        return {
            "feedback": "Unable to query quant critic - API unavailable",
            "improved_rationale": None,
            "risk_assessment": None,
            "is_optimal": False,
        }

    try:
        result = json.loads(response_text)
        # Ensure all expected keys are present
        return {
            "feedback": result.get("feedback", ""),
            "improved_rationale": result.get("improved_rationale"),
            "risk_assessment": result.get("risk_assessment"),
            "is_optimal": result.get("is_optimal", False),
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing quant critic response: {e}")
        # Return the raw text as feedback if JSON parsing fails
        return {
            "feedback": response_text,
            "improved_rationale": None,
            "risk_assessment": None,
            "is_optimal": False,
        }


def improve_trading_rationale(
    ticker: str,
    timeframe: str,
    prediction: float,
    confidence: float,
    last_price: float,
    features: dict = None,
    gate_results: dict = None,
    model: str = "gpt-4",
) -> dict:
    """Generate improved trading rationale using the quant critic.

    This is a convenience function that constructs the appropriate inputs
    for the quant critic from common trading signal components.

    Args:
        ticker: Trading instrument
        timeframe: Prediction timeframe
        prediction: Model prediction value
        confidence: Model confidence score (0-1 or 0-100)
        last_price: Last observed price
        features: Dict of key features used in prediction (optional)
        gate_results: Results from trading gates (optional)
        model: OpenAI model to use

    Returns:
        Dict with improved rationale and feedback
    """
    # Construct buddy_raw from inputs
    buddy_raw = {
        "prediction": prediction,
        "confidence": confidence,
        "last_price": last_price,
        "delta": prediction - last_price,
        "direction": "long" if prediction > last_price else "short",
    }

    if features:
        buddy_raw["key_features"] = features
    if gate_results:
        buddy_raw["gate_results"] = gate_results

    # Construct current response
    delta = prediction - last_price
    direction = "BUY" if delta > 0 else "SELL"
    confidence_str = f"{confidence:.1%}" if confidence <= 1 else f"{confidence:.1f}"
    current_response = (
        f"Signal: {direction} {ticker}\n"
        f"Prediction: {prediction:.5f}, Last: {last_price:.5f}, Delta: {delta:+.5f}\n"
        f"Confidence: {confidence_str}"
    )

    return query_quant_critic(
        ticker=ticker,
        timeframe=timeframe,
        buddy_raw=buddy_raw,
        current_response=current_response,
        model=model,
    )
