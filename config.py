from pathlib import Path

from platformdirs import user_cache_path, user_data_path, user_log_path


APPNAME: str = "imbalance-multi-classification"
AUTHOR: str = "Muhammad Rajaul Ghufran"

APPCACHE_PATH: Path = user_cache_path(APPNAME, AUTHOR)
APPDATA_PATH: Path = user_data_path(APPNAME, AUTHOR)
APPLOG_PATH: Path = user_log_path(APPNAME, AUTHOR)