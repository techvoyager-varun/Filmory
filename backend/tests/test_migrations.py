from pathlib import Path
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from app.database import Base

def test_fresh_migrations_create_every_application_table(tmp_path):
    backend=Path(__file__).resolve().parents[1]
    config=Config(str(backend/'alembic.ini'))
    config.set_main_option('script_location',str(backend/'migrations'))
    engine=create_engine('sqlite:///'+(tmp_path/'migrations.sqlite').as_posix())
    with engine.begin() as connection:
        config.attributes['connection']=connection
        command.upgrade(config,'head')
        inspector=inspect(connection)
        assert set(Base.metadata.tables) <= set(inspector.get_table_names())
        for name,table in Base.metadata.tables.items():
            assert set(table.columns.keys()) == {c['name'] for c in inspector.get_columns(name)}
    engine.dispose()
