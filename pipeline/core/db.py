"""
pipeline/core/db.py

SQLite engine setup, session factory, and DB initializer.
Import `engine` and `get_session` anywhere DB access is needed.
"""

import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)

# Database file sits at project root
DB_PATH = Path(__file__).parent.parent.parent / "ollama.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


def get_session():
    """Yield a DB session — use as a context manager or dependency."""
    with Session(engine) as session:
        yield session


def init_db():
    """Create all tables if they don't exist yet."""
    logger.info(f"Initializing DB at {DB_PATH}")
    SQLModel.metadata.create_all(engine)
    logger.info("DB ready.")
