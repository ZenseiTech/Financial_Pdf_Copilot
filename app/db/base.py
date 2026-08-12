from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models using 2.0 DeclarativeBase syntax.
    Custom mixins, tablename generators, or common columns (e.g., created_at)
    can be defined here.
    """
    pass