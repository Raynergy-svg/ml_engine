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


def set_openai_credentials(config: dict) -> None:
    """Set OpenAI credentials from config or environment variables."""
    openai.api_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.error("OpenAI API key not found in configuration or .env")
        raise ValueError("Missing OpenAI API key")


client = openai


def query_openai(prompt, model="gpt-4", temperature=0.2, max_attempts=3):
    """General-purpose function to query the OpenAI API."""
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
        logging.error(f"Failed to parse GPT response: {e}\nResponse was: {response}")
        config_updates = yaml.safe_load(response)
    except Exception as e:
        logging.error(f"Failed to parse GPT response: {e}\nResponse was: {response}")
        config_updates = {}

    logging.debug(f"Config Updates:\n{config_updates}")
    return config_updates


def report_config_changes(current_config, config_updates):
    """Merge config with updates and write to config.yaml."""
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

    with open("config.yaml", "w") as f:
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
    logging.info(f"Autotuned configuration: {tuned_params}")
    return config
