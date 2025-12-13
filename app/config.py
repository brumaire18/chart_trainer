import os
from pathlib import Path

# プロジェクトのルートディレクトリ
BASE_DIR = Path(__file__).resolve().parents[1]

# CSV保存場所
PRICE_CSV_DIR = BASE_DIR / "data" / "price_csv"

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
