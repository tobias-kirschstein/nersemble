from pathlib import Path

from environs import Env

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/nersemble/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("NERSEMBLE_"):
    NERSEMBLE_DATA_PATH = env("DATA_PATH")
    NERSEMBLE_MODELS_PATH = env("MODELS_PATH")
    NERSEMBLE_RENDERS_PATH = env("RENDERS_PATH")
