import configparser
import os
import rospy

def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _cfg_get(cfg: configparser.SectionProxy, key: str, default, cast):
    raw = cfg.get(key, fallback=None)
    if raw is None or raw == "":
        return default
    try:
        if cast is bool:
            return _as_bool(raw)
        return cast(raw)
    except Exception as exc:
        raise ValueError(f"Invalid config value for '{key}': {raw}") from exc


def _load_runtime_config() -> configparser.SectionProxy:
    default_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.cfg")
    config_path = os.environ.get("OBJECTNAV_CONFIG", default_cfg_path)

    parser = configparser.ConfigParser()
    loaded = parser.read(config_path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(
            f"Cannot read config file: {config_path}. "
            "Set OBJECTNAV_CONFIG or create app/config.cfg."
        )
    if "objectnav" not in parser:
        raise KeyError(
            f"Missing [objectnav] section in config file: {config_path}"
        )
    rospy.loginfo("Loaded runtime config: %s", config_path)
    return parser["objectnav"]