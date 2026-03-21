from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config import META_DIR, PRICE_CSV_DIR


STYLE_BUCKET_CYCLICAL = "cyclical"
STYLE_BUCKET_DEFENSIVE = "defensive"
STYLE_BUCKET_NEUTRAL = "neutral"
VALID_STYLE_BUCKETS = {
    STYLE_BUCKET_CYCLICAL,
    STYLE_BUCKET_DEFENSIVE,
    STYLE_BUCKET_NEUTRAL,
}

LEADLAG_UNIVERSE_COLUMNS = [
    "symbol",
    "market",
    "name",
    "sector",
    "style_bucket",
    "path_group",
]

LEADLAG_UNIVERSE_CSV_PATH = META_DIR / "leadlag_universe.csv"
LEADLAG_UNIVERSE_JSON_PATH = META_DIR / "leadlag_universe.json"

LEADLAG_US_PRICE_DIR = PRICE_CSV_DIR / "leadlag_us"
LEADLAG_JP_PRICE_DIR = PRICE_CSV_DIR / "leadlag_jp"


@dataclass(frozen=True)
class LeadLagUniverseItem:
    symbol: str
    market: str
    name: str
    sector: str
    style_bucket: str
    path_group: str


_US_11_SECTOR_ETFS: List[LeadLagUniverseItem] = [
    LeadLagUniverseItem("XLB", "US", "Materials Select Sector SPDR Fund", "Materials", "cyclical", "materials"),
    LeadLagUniverseItem("XLC", "US", "Communication Services Select Sector SPDR Fund", "Communication Services", "neutral", "communication_services"),
    LeadLagUniverseItem("XLE", "US", "Energy Select Sector SPDR Fund", "Energy", "cyclical", "energy"),
    LeadLagUniverseItem("XLF", "US", "Financial Select Sector SPDR Fund", "Financials", "cyclical", "financials"),
    LeadLagUniverseItem("XLI", "US", "Industrial Select Sector SPDR Fund", "Industrials", "cyclical", "industrials"),
    LeadLagUniverseItem("XLK", "US", "Technology Select Sector SPDR Fund", "Information Technology", "cyclical", "information_technology"),
    LeadLagUniverseItem("XLP", "US", "Consumer Staples Select Sector SPDR Fund", "Consumer Staples", "defensive", "consumer_staples"),
    LeadLagUniverseItem("XLRE", "US", "Real Estate Select Sector SPDR Fund", "Real Estate", "neutral", "real_estate"),
    LeadLagUniverseItem("XLU", "US", "Utilities Select Sector SPDR Fund", "Utilities", "defensive", "utilities"),
    LeadLagUniverseItem("XLV", "US", "Health Care Select Sector SPDR Fund", "Health Care", "defensive", "health_care"),
    LeadLagUniverseItem("XLY", "US", "Consumer Discretionary Select Sector SPDR Fund", "Consumer Discretionary", "cyclical", "consumer_discretionary"),
]


_JP_TOPIX17_ETFS: List[LeadLagUniverseItem] = [
    LeadLagUniverseItem("1617", "JP", "NEXT FUNDS TOPIX-17 FOODS ETF", "Foods", "defensive", "consumer_staples"),
    LeadLagUniverseItem("1618", "JP", "NEXT FUNDS TOPIX-17 ENERGY RESOURCES ETF", "Energy Resources", "cyclical", "energy"),
    LeadLagUniverseItem("1619", "JP", "NEXT FUNDS TOPIX-17 CONSTRUCTION AND MATERIALS ETF", "Construction and Materials", "cyclical", "materials"),
    LeadLagUniverseItem("1620", "JP", "NEXT FUNDS TOPIX-17 RAW MATERIALS AND CHEMICALS ETF", "Raw Materials and Chemicals", "cyclical", "materials"),
    LeadLagUniverseItem("1621", "JP", "NEXT FUNDS TOPIX-17 PHARMACEUTICAL ETF", "Pharmaceutical", "defensive", "health_care"),
    LeadLagUniverseItem("1622", "JP", "NEXT FUNDS TOPIX-17 AUTOMOBILES AND TRANSPORTATION EQUIPMENT ETF", "Automobiles and Transportation Equipment", "cyclical", "consumer_discretionary"),
    LeadLagUniverseItem("1623", "JP", "NEXT FUNDS TOPIX-17 STEEL AND NONFERROUS METALS ETF", "Steel and Nonferrous Metals", "cyclical", "materials"),
    LeadLagUniverseItem("1624", "JP", "NEXT FUNDS TOPIX-17 MACHINERY ETF", "Machinery", "cyclical", "industrials"),
    LeadLagUniverseItem("1625", "JP", "NEXT FUNDS TOPIX-17 ELECTRIC APPLIANCES AND PRECISION INSTRUMENTS ETF", "Electric Appliances and Precision Instruments", "cyclical", "information_technology"),
    LeadLagUniverseItem("1626", "JP", "NEXT FUNDS TOPIX-17 IT AND SERVICES OTHER ETF", "IT and Services, Others", "neutral", "communication_services"),
    LeadLagUniverseItem("1627", "JP", "NEXT FUNDS TOPIX-17 ELECTRIC POWER AND GAS ETF", "Electric Power and Gas", "defensive", "utilities"),
    LeadLagUniverseItem("1628", "JP", "NEXT FUNDS TOPIX-17 TRANSPORTATION AND LOGISTICS ETF", "Transportation and Logistics", "cyclical", "industrials"),
    LeadLagUniverseItem("1629", "JP", "NEXT FUNDS TOPIX-17 COMMERCIAL AND WHOLESALE TRADE ETF", "Commercial and Wholesale Trade", "cyclical", "industrials"),
    LeadLagUniverseItem("1630", "JP", "NEXT FUNDS TOPIX-17 RETAIL TRADE ETF", "Retail Trade", "neutral", "consumer_discretionary"),
    LeadLagUniverseItem("1631", "JP", "NEXT FUNDS TOPIX-17 BANKS ETF", "Banks", "cyclical", "financials"),
    LeadLagUniverseItem("1632", "JP", "NEXT FUNDS TOPIX-17 FINANCIALS EX BANKS ETF", "Financials (ex Banks)", "cyclical", "financials"),
    LeadLagUniverseItem("1633", "JP", "NEXT FUNDS TOPIX-17 REAL ESTATE ETF", "Real Estate", "cyclical", "real_estate"),
]


def ensure_leadlag_directories() -> None:
    LEADLAG_US_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    LEADLAG_JP_PRICE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_leadlag_symbol(symbol: str) -> str:
    raw = str(symbol).strip()
    if not raw:
        raise ValueError("symbol must be non-empty")
    if raw.isdigit():
        return raw.zfill(4)
    return raw.upper()


def normalize_leadlag_market(market: Optional[str]) -> Optional[str]:
    if market is None:
        return None
    normalized = str(market).strip().upper()
    if not normalized:
        return None
    if normalized not in {"US", "JP"}:
        raise ValueError("market must be one of: US, JP")
    return normalized


def get_leadlag_price_dir(market: str) -> Path:
    normalized = normalize_leadlag_market(market)
    if normalized == "US":
        return LEADLAG_US_PRICE_DIR
    if normalized == "JP":
        return LEADLAG_JP_PRICE_DIR
    raise ValueError("market must be one of: US, JP")


def _validate_item(item: LeadLagUniverseItem) -> LeadLagUniverseItem:
    symbol = normalize_leadlag_symbol(item.symbol)
    market = normalize_leadlag_market(item.market)
    if market is None:
        raise ValueError("market must be one of: US, JP")
    style_bucket = str(item.style_bucket).strip().lower()
    if style_bucket not in VALID_STYLE_BUCKETS:
        raise ValueError(
            "style_bucket must be one of: cyclical, defensive, neutral"
        )
    return LeadLagUniverseItem(
        symbol=symbol,
        market=market,
        name=str(item.name).strip(),
        sector=str(item.sector).strip(),
        style_bucket=style_bucket,
        path_group=str(item.path_group).strip(),
    )


def _records_to_items(records: Sequence[Dict[str, object]]) -> List[LeadLagUniverseItem]:
    items: List[LeadLagUniverseItem] = []
    for record in records:
        item = LeadLagUniverseItem(
            symbol=str(record.get("symbol", "")),
            market=str(record.get("market", "")),
            name=str(record.get("name", "")),
            sector=str(record.get("sector", "")),
            style_bucket=str(record.get("style_bucket", "")),
            path_group=str(record.get("path_group", "")),
        )
        items.append(_validate_item(item))
    return sorted(items, key=lambda x: (x.market, x.symbol))


def _items_to_frame(items: Sequence[LeadLagUniverseItem]) -> pd.DataFrame:
    normalized_items = [_validate_item(item) for item in items]
    records = [asdict(item) for item in normalized_items]
    df = pd.DataFrame(records, columns=LEADLAG_UNIVERSE_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=LEADLAG_UNIVERSE_COLUMNS)
    return df.sort_values(["market", "symbol"]).reset_index(drop=True)


def get_default_leadlag_universe_items(
    market: Optional[str] = None,
) -> List[LeadLagUniverseItem]:
    normalized_market = normalize_leadlag_market(market)
    items = list(_US_11_SECTOR_ETFS) + list(_JP_TOPIX17_ETFS)
    items = [_validate_item(item) for item in items]
    if normalized_market:
        items = [item for item in items if item.market == normalized_market]
    return sorted(items, key=lambda x: (x.market, x.symbol))


def save_leadlag_universe_csv(
    items: Optional[Sequence[LeadLagUniverseItem]] = None,
    path: Path = LEADLAG_UNIVERSE_CSV_PATH,
) -> Path:
    target_items = list(items) if items is not None else get_default_leadlag_universe_items()
    df = _items_to_frame(target_items)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_leadlag_universe_csv(path: Path = LEADLAG_UNIVERSE_CSV_PATH) -> List[LeadLagUniverseItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("leadlag universe csv was not found: {}".format(path))
    df = pd.read_csv(path)
    missing = [col for col in LEADLAG_UNIVERSE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("leadlag universe csv is missing columns: {}".format(",".join(missing)))
    records = df[LEADLAG_UNIVERSE_COLUMNS].to_dict(orient="records")
    return _records_to_items(records)


def save_leadlag_universe_json(
    items: Optional[Sequence[LeadLagUniverseItem]] = None,
    path: Path = LEADLAG_UNIVERSE_JSON_PATH,
) -> Path:
    target_items = list(items) if items is not None else get_default_leadlag_universe_items()
    records = [asdict(_validate_item(item)) for item in target_items]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_leadlag_universe_json(path: Path = LEADLAG_UNIVERSE_JSON_PATH) -> List[LeadLagUniverseItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("leadlag universe json was not found: {}".format(path))
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("leadlag universe json must be an array of objects")
    normalized_records: List[Dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("leadlag universe json contains a non-object item")
        normalized_records.append(record)
    return _records_to_items(normalized_records)


def load_leadlag_universe(path: Optional[Path] = None) -> List[LeadLagUniverseItem]:
    if path is not None:
        suffix = str(path).lower()
        if suffix.endswith(".csv"):
            return load_leadlag_universe_csv(path)
        if suffix.endswith(".json"):
            return load_leadlag_universe_json(path)
        raise ValueError("path must end with .csv or .json")

    if LEADLAG_UNIVERSE_CSV_PATH.exists():
        return load_leadlag_universe_csv(LEADLAG_UNIVERSE_CSV_PATH)
    if LEADLAG_UNIVERSE_JSON_PATH.exists():
        return load_leadlag_universe_json(LEADLAG_UNIVERSE_JSON_PATH)
    return get_default_leadlag_universe_items()


def find_leadlag_item(
    symbol: str,
    universe_items: Optional[Sequence[LeadLagUniverseItem]] = None,
) -> Optional[LeadLagUniverseItem]:
    normalized_symbol = normalize_leadlag_symbol(symbol)
    items = (
        list(universe_items)
        if universe_items is not None
        else load_leadlag_universe()
    )
    for item in items:
        if normalize_leadlag_symbol(item.symbol) == normalized_symbol:
            return _validate_item(item)
    return None


def get_leadlag_universe_map(
    market: Optional[str] = None,
    universe_items: Optional[Sequence[LeadLagUniverseItem]] = None,
) -> Dict[str, LeadLagUniverseItem]:
    normalized_market = normalize_leadlag_market(market)
    items = (
        list(universe_items)
        if universe_items is not None
        else load_leadlag_universe()
    )
    result: Dict[str, LeadLagUniverseItem] = {}
    for item in items:
        normalized_item = _validate_item(item)
        if normalized_market and normalized_item.market != normalized_market:
            continue
        result[normalized_item.symbol] = normalized_item
    return result


ensure_leadlag_directories()
