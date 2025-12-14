import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """簡易的に .env を読み込み、環境変数に設定する。

    python-dotenv に頼らず、``KEY=VALUE`` 形式の行だけを読み込む。
    既に環境変数が設定済みの場合は上書きしない。
    """

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value

# プロジェクトのルートディレクトリ
BASE_DIR = Path(__file__).resolve().parents[1]

# .env を事前に取り込む
_load_dotenv(BASE_DIR / ".env")

# CSV保存場所
PRICE_CSV_DIR = BASE_DIR / "data" / "price_csv"

# メタ情報保存場所
META_DIR = BASE_DIR / "data" / "meta"

# DBファイル（フェーズ1ではまだ使わなくてもOK）
DB_DIR = BASE_DIR / "data" / "db"
DB_PATH = DB_DIR / "trainer.sqlite"

# API settings
JQUANTS_REFRESH_TOKEN = os.getenv("JQUANTS_REFRESH_TOKEN")
JQUANTS_BASE_URL = os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com")
JQUANTS_MAILADDRESS = os.getenv("MAILADDRESS")
JQUANTS_PASSWORD = os.getenv("PASSWORD")

# デフォルトの問題長
DEFAULT_LOOKBACK_BARS = 100  # 過去本数
DEFAULT_ANSWER_BARS = 40  # 答え本数

# Ensure required directories exist
PRICE_CSV_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
