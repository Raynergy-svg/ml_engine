try:
    import numpy as np

    print("NumPy imported successfully, version:", np.__version__)
except Exception as e:
    print("Failed to import NumPy:", e)

try:
    import torch

    print("Torch imported successfully, version:", torch.__version__)
except Exception as e:
    print("Failed to import Torch:", e)

try:
    from packaging import version

    print("Packaging imported successfully")
except Exception as e:
    print("Failed to import packaging:", e)

try:
    import pandas as pd

    print("Pandas imported successfully, version:", pd.__version__)
except Exception as e:
    print("Failed to import Pandas:", e)

try:
    import matplotlib.pyplot as plt

    print("Matplotlib imported successfully, version:", plt.__version__)
except Exception as e:
    print("Failed to import Matplotlib:", e)

try:
    import sklearn

    print("Scikit-learn imported successfully, version:", sklearn.__version__)
except Exception as e:
    print("Failed to import Scikit-learn:", e)

try:
    import optuna

    print("Optuna imported successfully, version:", optuna.__version__)
except Exception as e:
    print("Failed to import Optuna:", e)

# ...add additional imports as necessary...
