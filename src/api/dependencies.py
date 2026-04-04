"""FastAPI dependency providers for storage backends.

Stores are initialized once at application startup and stored on app.state.
Dependencies read from app.state so they can be overridden in tests by
passing custom instances to create_app().
"""
from fastapi import Request

from src.storage import Database, EventStore, FAISSStore


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_faiss_store(request: Request) -> FAISSStore:
    return request.app.state.faiss_store


def get_event_store(request: Request) -> EventStore:
    return request.app.state.event_store
