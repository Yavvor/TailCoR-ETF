from datetime import date

from pydantic import BaseModel


class AssetMetadata(BaseModel):
    """Przechowuje metadane o aktywie (zgodnie ze schematem)."""
    ticker: str
    name: str
    asset_type: str = "equity"  # np. etf, stock, synthetic
    market_sector: str
    data_start: date
    data_end: date
