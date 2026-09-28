try:
    from .cli import mole_predict
except Exception:
    mole_predict = None

try:
    from .cli import mole_train
except Exception:
    mole_train = None

__all__ = ["mole_predict", "mole_train"]
