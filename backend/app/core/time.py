"""UTC storage and API formatting for existing timestamp-without-timezone columns."""
from datetime import datetime, timezone

def utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def utc_isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()
