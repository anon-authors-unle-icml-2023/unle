from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional
from typing_extensions import Self


@dataclass
class Config:
    results_dir: Path

    @classmethod
    def create(cls, results_dir: Optional[str]) -> Self:
        assert results_dir is not None, "results_dir must be specified"
        return cls(Path(results_dir).absolute())

    

def get_config_file_path() -> str:
    current_dir = Path.cwd()
    max_depth = 50
    for _ in range(max_depth):
        config_file_path = current_dir / "experiments_utils_config.json"
        if config_file_path.exists():
            return str(config_file_path)
        if (current_dir / ".git").exists():
            raise FileNotFoundError(
                "Could not find config.json file. Please make sure you are running "
                "this script from a directory that contains a experiments_utils_config.json file."
            )
        current_dir = current_dir.parent
    else:
        raise FileNotFoundError("Exceeded max depth when looking for experiments_utils_config.json file")


def get_config() -> Config:
    config_file_path = get_config_file_path()
    with open(config_file_path, "r") as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return Config.create(results_dir=config.get("results_dir"))
