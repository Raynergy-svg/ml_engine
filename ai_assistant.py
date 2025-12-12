import logging
import ast
import subprocess
import tempfile
import shutil
import os
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, List


try:
    import openai

    openai_available = True
except ImportError:
    openai_available = False

try:
    from anthropic import Client

    anthropic_available = True
except ImportError:
    anthropic_available = False


class DataError(Exception):
    """Raised when there are issues with data loading or processing."""

    pass


class ModelError(Exception):
    """Raised when there are issues with model operations."""

    pass


class ai_assistant:
    """Provides AI-based code explanations and improvement suggestions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.openai_enabled = False
        self.anthropic_client = None
        self.memory: List[str] = []
        self.suggestion_cache: Dict[str, str] = {}
        self.active_model = None
        self.training_data = None
        self.context_window = 4096
        # Parameterization
        self.max_tokens = config.get("max_tokens", 1500)
        self.temperature = config.get("temperature", 0.4)
        # Setup directories
        sandbox_dir = config.get("SANDBOX_DIR", "sandbox")
        self.sandbox_dir = Path(sandbox_dir)
        self.sandbox_dir.mkdir(exist_ok=True)
        # Setup device
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_workers = config.get("num_workers", config.get("MAX_WORKERS", 4))
        self.pin_memory = config.get("pin_memory", torch.cuda.is_available())

        # Initialize OpenAI
        openai_api_key = config.get("OPENAI_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if openai_available and openai_api_key:
            openai.api_key = openai_api_key
            self.openai_enabled = True
            logging.debug("OpenAI initialized.")
        else:
            logging.warning("OpenAI key missing or openai not installed.")

        # Initialize Anthropic
        anthropic_api_key = config.get("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if anthropic_available and anthropic_api_key:
            self.anthropic_client = Client(api_key=anthropic_api_key)
            logging.debug("Anthropic client initialized.")
        else:
            logging.warning("Anthropic key missing or anthropic not installed.")

    def load_training_data(self, data_path: str) -> bool:
        """Load training data or model weights from file."""
        try:
            if not os.path.isfile(data_path):
                raise DataError(f"Data file not found: {data_path}")

            ext = os.path.splitext(data_path)[1].lower()

            if ext == ".pth":
                self.active_model = torch.load(data_path)
                logging.info(f"Loaded PyTorch model from {data_path}")
                return True
            elif ext == ".npz":
                import numpy as np

                self.active_model = np.load(data_path)
                logging.info(f"Loaded NumPy model from {data_path}")
                return True
            else:
                raise DataError(f"Unsupported file format: {ext}")

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise DataError(f"Failed to load data: {str(e)}")

    def train_model(self, model_name: str) -> bool:
        """Train a model using ml_engine if available."""
        try:
            if not hasattr(self, "ml_engine") or self.ml_engine is None:
                raise ModelError("ML Engine not initialized")

            if not self.training_data:
                raise DataError("No training data available")

            logging.info(f"Training model {model_name} using ML Engine")
            self.ml_engine.train(self.training_data, epochs=10, batch_size=32)
            self.active_model = self.ml_engine.model
            return True

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise ModelError(f"Model training failed: {str(e)}")

    def run_interactive(self) -> None:
        choice = input(
            "Do you want code improvements or continue with training? (Enter 'improvements' or 'training'): "
        )
        if choice.strip().lower() == "improvements":
            if self.anthropic_client:
                result = self.get_code_improvements_via_anthropic()
                print("Code Improvements from Anthropic:")
                print(result)
            else:
                print("Anthropic is unavailable.")
        else:
            print("Continuing with training...")

    def get_code_improvements_via_anthropic(self) -> str:
        return "Suggested improvements: Refactor the training loop for better error handling and optimize data preprocessing."

    def simulate_scenario(self) -> str:
        # Placeholder
        try:
            return "Simulation functionality is currently disabled."
        except Exception as e:
            return f"Error during simulation: {e}"

    def process_query(self, query: str, use_claude: bool = False) -> str:
        if query.strip().lower() == "simulate":
            return self.simulate_scenario()
        logging.debug("Prompting user for simulation parameters.")
        print(
            "AI Assistant: Which historical market data would you like to use? (Enter time period, assets, etc.)"
        )
        data_input = input("User: ")
        print(
            "AI Assistant: Which machine learning model do you prefer? (e.g. random forest, LSTM, reinforcement learning)"
        )
        model_input = input("User: ")
        print(
            "AI Assistant: What are your primary trading objectives? (e.g. maximizing Sharpe ratio, limiting drawdown)"
        )
        objectives_input = input("User: ")
        print(
            "AI Assistant: How would you like the simulation results presented? (e.g. visualizations, dashboard, summary)"
        )
        presentation_input = input("User: ")
        response_text = (
            f"Simulation parameters received:\n"
            f"- Historical Data: {data_input}\n"
            f"- ML Model: {model_input}\n"
            f"- Trading Objectives: {objectives_input}\n"
            f"- Presentation Format: {presentation_input}\n"
            "Proceeding with simulation setup...\n"
        )
        return response_text

    def suggest_improvements(self, code_snippet: str) -> str:
        if code_snippet in self.suggestion_cache:
            return self.suggestion_cache[code_snippet]
        prompt = (
            "Review this ML engine code and suggest specific improvements:\n\n"
            f"{code_snippet}\n\n"
            "Focus on:\n- Performance optimizations\n- Best practices\n- Error handling\n- Memory efficiency"
        )
        try:
            if self.anthropic_client:
                resp = self.anthropic_client.completions.create(
                    model="claude-2",
                    prompt=prompt,
                    max_tokens_to_sample=self.max_tokens,
                )
                self.suggestion_cache[code_snippet] = resp.completion
                return resp.completion
            if self.openai_enabled:
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                )
                suggestion = resp.choices[0].message.content
                self.suggestion_cache[code_snippet] = suggestion
                return suggestion
            return "No AI client configured."
        except Exception as e:
            return f"Error suggesting improvements: {e}"

    def validate_code(self, code_snippet: str) -> bool:
        try:
            ast.parse(code_snippet)
            return True
        except SyntaxError:
            return False

    def execute_in_sandbox(
        self, code_snippet: str, timeout: int = 5
    ) -> Tuple[bool, str]:
        temp_dir = tempfile.mkdtemp(dir=self.sandbox_dir)
        script_path = Path(temp_dir) / "temp_script.py"
        try:
            with open(script_path, "w") as f:
                f.write(code_snippet)
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        except Exception as e:
            return False, str(e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_code_fix(self, code_snippet: str, error_message: str) -> str:
        prompt = (
            "You are an AI code assistant. The following code snippet produced an error:\n\n"
            + "```python\n"
            + code_snippet
            + "\n```\n\n"
            + "The error message is:\n\n"
            + error_message
            + "\n\n"
            + "Generate a corrected version of the code snippet that fixes the error. Focus solely on providing the corrected code. Do not include any explanations or surrounding text. Ensure the corrected code is syntactically correct and addresses the error."
        )
        try:
            if self.anthropic_client:
                try:
                    resp = self.anthropic_client.completions.create(
                        model="claude-3-opus-20240229",
                        prompt=prompt,
                        max_tokens_to_sample=self.max_tokens,
                        temperature=0.2,
                    )
                    return resp.completion.strip()
                except Exception as e:
                    logging.warning(
                        f"Anthropic model failed, falling back to OpenAI: {e}"
                    )
                    if self.openai_enabled:
                        fallback = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=self.max_tokens,
                        )
                        return fallback.choices[0].message.content.strip()
                    return "Error generating code fix: Anthropic failed and OpenAI not configured"
            if self.openai_enabled:
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content.strip()
            return "Error generating code fix: No AI client configured"
        except Exception as e:
            return f"Error generating code fix: {e}"

    def generate_code_fix_via_api(self, method_code: str, error_message: str) -> str:
        try:
            return self.generate_code_fix(method_code, error_message)
        except Exception as e:
            logging.error(f"Error generating code fix via API: {e}")
            return ""

    def validate_code_via_api(self, code_fix: str) -> Tuple[bool, str]:
        try:
            import requests

            api_url = self.config.get(
                "api_execute_sandbox_url", "http://example.com/api/execute_in_sandbox"
            )
            response = requests.post(api_url, json={"code": code_fix})
            response.raise_for_status()
            data = response.json()
            return data.get("success", False), data.get("output", "")
        except Exception as e:
            logging.error(f"Error executing code fix in sandbox via API: {e}")
            return False, str(e)

    def advanced_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run advanced trading simulation with the active model."""
        try:
            if not self.active_model:
                raise ModelError("No trained model available for simulation")

            if not hasattr(self.active_model, "predict"):
                raise ModelError("Model does not support predictions")

            logging.info(f"Running simulation with parameters: {parameters}")

            # Example simulation logic
            results = {
                "success": True,
                "predictions": [],
                "metrics": {"accuracy": 0.0, "profit_loss": 0.0, "max_drawdown": 0.0},
                "message": f"Simulation complete using {type(self.active_model).__name__}",
            }

            return results

        except Exception as e:
            logging.error(f"Simulation failed: {str(e)}")
            return {
                "success": False,
                "message": f"Simulation failed: {str(e)}",
                "error": str(e),
            }

    def run_as_cli(self) -> None:
        import argparse

        parser = argparse.ArgumentParser(description="AI Assistant CLI")
        parser.add_argument("--simulate", action="store_true", help="Run a simulation.")
        parser.add_argument(
            "--train",
            nargs=1,
            metavar="MODEL",
            help="Train a model with the given name.",
        )
        parser.add_argument(
            "--load_data", nargs=1, metavar="FILE", help="Load data from a file."
        )
        args = parser.parse_args()

        if args.load_data:
            success = self.load_training_data(args.load_data[0])
            logging.info(f"Data load success: {success}")
        if args.train:
            trained = self.train_model(args.train[0])
            logging.info(f"Training success: {trained}")
        if args.simulate:
            print(self.simulate_scenario())


def run_tests():
    """Test the AI assistant with proper data processing."""
    assistant = ai_assistant({})

    # Test 1: Simulate scenario
    sim_result = assistant.process_query("simulate")
    assert "Simulation functionality is currently disabled." in sim_result

    # Test 2: Load model file - tests DataError handling
    try:
        assistant.load_training_data("./trained_data/models/best_model.pth")
        assert False, "Should have raised DataError"
    except DataError as e:
        assert "Data file not found" in str(e)

    # Test 3: Test ModelError handling for training without ML engine
    try:
        assistant.train_model("test_model")
        assert False, "Should have raised ModelError"
    except ModelError as e:
        assert "ML Engine not initialized" in str(e)

    # Test 4: Test advanced simulation without model
    adv_result = assistant.advanced_simulation({"strategy": "long"})
    assert adv_result["success"] is False
    assert "Simulation failed" in adv_result["message"]

    # Test 5: Test file format validation
    try:
        assistant.load_training_data("./market_data/invalid.pth")
        assert False, "Should have raised DataError"
    except DataError as e:
        assert "Unsupported file format" in str(e)

    logging.info("All tests completed successfully")


if __name__ == "__main__":
    run_tests()
