import os
import time
from pathlib import Path

from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast


def _get_cache_dir():
    """
    Use project-local HuggingFace cache by default:
    <repo_root>/cache/huggingface/hub
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_hf_home = os.path.join(repo_root, "cache", "huggingface")

    env_hf_home = os.environ.get("HF_HOME", "").strip()
    env_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()

    if env_hub_cache:
        hub_cache = env_hub_cache
        hf_home = env_hf_home or str(Path(env_hub_cache).parent)
    else:
        hf_home = env_hf_home or default_hf_home
        # Tolerate common misconfiguration: HF_HOME points directly to ".../hub".
        if Path(hf_home).name == "hub":
            hub_cache = hf_home
            hf_home = str(Path(hf_home).parent)
        else:
            hub_cache = os.path.join(hf_home, "hub")

    # Persist normalized paths for this process so downstream calls share the same roots.
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hub_cache

    os.makedirs(hub_cache, exist_ok=True)
    return hub_cache


def _looks_like_local_model_dir(path: str) -> bool:
    """Check whether path contains a usable local text encoder."""
    p = Path(path)
    if not p.is_dir():
        return False

    has_config = (p / "config.json").exists()
    has_tokenizer = (p / "tokenizer_config.json").exists() or (p / "tokenizer.json").exists()
    has_vocab = (p / "vocab.txt").exists()
    has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    return has_config and has_tokenizer and has_vocab and has_weights


def _auto_resolve_local_text_encoder(text_encoder_type: str) -> str:
    """
    Auto-resolve local model directories for migration/offline scenarios.
    Priority:
    1) direct path in text_encoder_type
    2) <repo_root>/models/<text_encoder_type>
    3) latest snapshot in HuggingFace hub cache
    """
    if os.path.isdir(text_encoder_type):
        return text_encoder_type

    repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    candidate = repo_root / "models" / text_encoder_type
    if _looks_like_local_model_dir(str(candidate)):
        return str(candidate)

    hub_cache = Path(_get_cache_dir())
    model_key = text_encoder_type.replace("/", "--")
    snapshots_root = hub_cache / f"models--{model_key}" / "snapshots"
    if snapshots_root.is_dir():
        snapshot_dirs = [d for d in snapshots_root.iterdir() if d.is_dir()]
        snapshot_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for snap in snapshot_dirs:
            if _looks_like_local_model_dir(str(snap)):
                return str(snap)

    return text_encoder_type


def _resolve_text_encoder_type(text_encoder_type):
    # Allow users to force a local model directory to avoid network downloads.
    override = os.environ.get("GROUNDINGDINO_TEXT_ENCODER_PATH", "").strip()
    if override:
        return override
    if isinstance(text_encoder_type, str):
        return _auto_resolve_local_text_encoder(text_encoder_type)
    return text_encoder_type


def _network_like_error(exc: Exception) -> bool:
    msg = str(exc)
    patterns = (
        "InvalidChunkLength",
        "ChunkedEncodingError",
        "Connection broken",
        "Read timed out",
        "Connection reset by peer",
        "Temporary failure in name resolution",
        "RemoteDisconnected",
        "ProtocolError",
    )
    return any(p in msg for p in patterns)


def _from_pretrained_with_retry(loader_cls, text_encoder_type, retries=3):
    offline = (
        os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    )
    force_download = os.environ.get("GROUNDINGDINO_FORCE_DOWNLOAD", "0") == "1"
    cache_dir = _get_cache_dir()
    last_error = None

    # 1) Prefer local cache first to avoid a network request at every startup.
    if not force_download:
        try:
            return loader_cls.from_pretrained(
                text_encoder_type,
                local_files_only=True,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            last_error = exc
            if offline:
                raise RuntimeError(
                    "Offline mode is enabled, but local text encoder is missing/incomplete. "
                    f"text_encoder_type={text_encoder_type}."
                ) from exc

    # 2) Local cache miss: fallback to network with retry.
    for attempt in range(1, retries + 1):
        try:
            return loader_cls.from_pretrained(
                text_encoder_type,
                local_files_only=False,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries or not _network_like_error(exc):
                break
            wait_s = attempt
            print(
                f"[GroundingDINO] failed to download '{text_encoder_type}' "
                f"(attempt {attempt}/{retries}): {exc}. Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)

    raise RuntimeError(
        "Failed to load text encoder. "
        f"text_encoder_type={text_encoder_type}. "
        f"cache_dir={cache_dir}. "
        "Set GROUNDINGDINO_TEXT_ENCODER_PATH to a local model directory "
        "or pre-download the model to HuggingFace cache."
    ) from last_error

def get_tokenlizer(text_encoder_type):
    text_encoder_type = _resolve_text_encoder_type(text_encoder_type)
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    if os.environ.get("GROUNDINGDINO_VERBOSE", "0") == "1":
        print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = _from_pretrained_with_retry(AutoTokenizer, text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    text_encoder_type = _resolve_text_encoder_type(text_encoder_type)
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return _from_pretrained_with_retry(BertModel, text_encoder_type)
    if text_encoder_type == "roberta-base":
        return _from_pretrained_with_retry(RobertaModel, text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
