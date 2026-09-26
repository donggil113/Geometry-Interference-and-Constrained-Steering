import os
from pathlib import Path

# Keep model/dataset caches outside the repository.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[3] / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
