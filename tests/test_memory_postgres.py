"""
Unit tests for PostgreSQL Conversational Memory Subsystem.
"""

import pytest
from app.storage.postgres import PostgresManager


@pytest.fixture
def in_memory_db():
    db = PostgresManager(db_url="sqlite:///:memory:")
    return db


def test_conversation_creation(in_memory_db):
    conv_id = in_memory_db.get_or_create_conversation()
    assert conv_id is not None
    assert len(conv_id) > 0

    # Retrieve existing
    same_id = in_memory_db.get_or_create_conversation(conv_id)
    assert same_id == conv_id


def test_message_persistence_and_ordering(in_memory_db):
    conv_id = in_memory_db.get_or_create_conversation()

    in_memory_db.add_message(conv_id, role="user", content="Hello", intent="casual_query")
    in_memory_db.add_message(conv_id, role="assistant", content="Hi! How can I help?", intent="casual_query")
    in_memory_db.add_message(conv_id, role="user", content="What is the attendance rule?", intent="knowledge_query")

    history = in_memory_db.get_conversation_history(conv_id)
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[0].content == "Hello"
    assert history[1].role == "assistant"
    assert history[2].role == "user"
    assert history[2].content == "What is the attendance rule?"


def test_recent_messages_sliding_window(in_memory_db):
    conv_id = in_memory_db.get_or_create_conversation()

    for i in range(10):
        in_memory_db.add_message(conv_id, role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")

    # Fetch last 4 messages
    recent = in_memory_db.get_recent_messages(conv_id, limit=4)
    assert len(recent) == 4
    # Must be in chronological order: 6, 7, 8, 9
    assert recent[0]["content"] == "Message 6"
    assert recent[3]["content"] == "Message 9"
