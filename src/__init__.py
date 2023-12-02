from src.model import GPT
from src.config import Config
from src.tokenizer import Tokenizer
from lightning_utilities.core.imports import RequirementCache

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.1.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        "Lit-GPT requires Lightning nightly (future lightning 2.1). Please run:\n"
        f" pip uninstall -y lightning; pip install -r requirements.txt\n{str(_LIGHTNING_AVAILABLE)}"
    )


__all__ = ["GPT", "Config", "Tokenizer"]
