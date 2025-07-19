import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def validate_path(path: str, create: bool = True) -> bool:
    path_folders = path.rstrip("/").split("/")
    cum_path = ""
    for folder in path_folders:
        cum_path += f"/{folder}" if cum_path != "" else folder
        if not os.path.exists(cum_path):
            if create:
                logging.info(f"Path {cum_path} not found. Creating directory {folder}")
                os.mkdir(cum_path)
            else:
                logging.info(f"Path {cum_path} not found. Returning False")
                return False
    return True
