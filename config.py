from dataclasses import dataclass, asdict
import json
from pathlib import Path

CFG_PATH = Path("config.json")

@dataclass
class Config:
    MODEL_PATH: str = "./train/weights/best.pt"

    IMG_SIZE: int = 640

    WINDOW_TITLE: str = None

    CONFIDENCE: float = 0.6

    PLAYER_RADIUS: float = 150

    def save(self, path: Path = CFG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path = CFG_PATH):
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))
