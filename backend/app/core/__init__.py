"""Core helpers, loaded lazily so model defaults do not import auth dependencies."""

def __getattr__(name):
    if name in {"verify_password", "get_password_hash", "create_access_token", "decode_access_token"}:
        from app.core import security
        return getattr(security, name)
    if name in {"get_current_user", "get_optional_current_user"}:
        from app.core import deps
        return getattr(deps, name)
    raise AttributeError(name)
