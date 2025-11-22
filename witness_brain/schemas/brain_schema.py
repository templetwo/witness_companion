# schemas/brain_schema.py - Brain output schema
from dataclasses import dataclass
from typing import Optional

@dataclass
class BrainOutput:
    """Represents the output from the Brain's thought process."""
    thought: str          # internal reflection / interpretation
    speech: str           # what Witness will actually say out loud
    action: Optional[str] = None  # e.g., "log_mood_observation", "speak"

    def __str__(self):
        action_str = f", action={self.action}" if self.action else ""
        return f"BrainOutput(thought='{self.thought[:50]}...', speech='{self.speech[:50]}...'{action_str})"
