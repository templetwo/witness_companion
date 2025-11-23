from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Event:
    """Represents a single sensory event in the Witness system."""
    source: str  # e.g., "stt", "vision", "system"
    content: str # The actual data, e.g., transcribed text
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self):
        return f"[{self.timestamp.isoformat()}] [{self.source.upper()}] {self.content}"

