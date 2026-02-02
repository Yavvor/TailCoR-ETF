from pydantic import BaseModel, Field
from datetime import date, datetime
class MarketConfig(BaseModel):
    """Konfiguracja Rynku (zgodnie ze schematem 'Market')."""
    start_date: date
    end_date: date
    base_currency: str = "PLN"
    risk_free_rate: float = 0.02