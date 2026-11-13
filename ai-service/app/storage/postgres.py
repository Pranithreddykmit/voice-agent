"""
PostgreSQL Storage and Conversational Memory Subsystem.
Manages persistent conversations and chat message history using SQLAlchemy.
Includes graceful fallback for lightweight testing or offline operation.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    select
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from app.config import settings
from app.schemas import MessageSchema

logger = logging.getLogger(__name__)

Base = declarative_base()


class ConversationModel(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan", order_by="MessageModel.created_at")


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    intent = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    conversation = relationship("ConversationModel", back_populates="messages")


class PostgresManager:
    """Manages database connection, session lifecycle, and conversation persistence."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self._engine = None
        self._session_factory = None
        self._initialize_engine()

    def _initialize_engine(self):
        try:
            self._engine = create_engine(
                self.db_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
            Base.metadata.create_all(self._engine)
            logger.info(f"Connected to PostgreSQL database at {self.db_url.split('@')[-1] if '@' in self.db_url else self.db_url}")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL ({str(e)}). Initializing SQLite in-memory fallback for memory.")
            self._engine = create_engine("sqlite:///:memory:", echo=False)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
            Base.metadata.create_all(self._engine)

    def is_connected(self) -> bool:
        """Check whether the database is connected and responsive."""
        if not self._engine:
            return False
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Retrieves an existing conversation ID or creates a new one.
        """
        with self._session_factory() as session:
            if conversation_id:
                conv = session.get(ConversationModel, conversation_id)
                if conv:
                    return conv.id

            new_id = conversation_id or str(uuid.uuid4())
            conv = ConversationModel(id=new_id)
            session.add(conv)
            session.commit()
            logger.info(f"Created new conversation session: {new_id}")
            return new_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None
    ) -> MessageSchema:
        """
        Appends a message to the conversation history.
        """
        with self._session_factory() as session:
            # Ensure conversation exists
            conv = session.get(ConversationModel, conversation_id)
            if not conv:
                conv = ConversationModel(id=conversation_id)
                session.add(conv)
                session.flush()

            msg = MessageModel(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role=role,
                content=content,
                intent=intent,
                created_at=datetime.now(timezone.utc)
            )
            session.add(msg)
            conv.updated_at = datetime.now(timezone.utc)
            session.commit()

            return MessageSchema(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                content=msg.content,
                intent=msg.intent,
                created_at=msg.created_at.isoformat()
            )

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 6
    ) -> List[Dict[str, str]]:
        """
        Retrieves the most recent messages for the LLM context window.
        Returns a list of dicts: [{"role": "user", "content": "..."}, ...]
        """
        with self._session_factory() as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.conversation_id == conversation_id)
                .order_by(MessageModel.created_at.desc())
                .limit(limit)
            )
            messages = session.scalars(stmt).all()
            # Reverse so they are in chronological order
            ordered = list(reversed(messages))
            return [{"role": m.role, "content": m.content} for m in ordered]

    def get_conversation_history(
        self,
        conversation_id: str
    ) -> List[MessageSchema]:
        """
        Retrieves the complete message history for a conversation.
        """
        with self._session_factory() as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.conversation_id == conversation_id)
                .order_by(MessageModel.created_at.asc())
            )
            messages = session.scalars(stmt).all()
            return [
                MessageSchema(
                    id=m.id,
                    conversation_id=m.conversation_id,
                    role=m.role,
                    content=m.content,
                    intent=m.intent,
                    created_at=m.created_at.isoformat()
                )
                for m in messages
            ]
