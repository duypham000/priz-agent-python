import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.persistence.models.base import Base


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    thread_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_name: Mapped[str] = mapped_column(String(500), nullable=False)
    deadline: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    owner: Mapped[str] = mapped_column(String(255), nullable=False, default="Unassigned")
    is_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
