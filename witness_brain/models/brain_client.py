# models/brain_client.py - Brain client for LLM integration
import json
import logging
import re
import requests
import yaml
from typing import List

from witness_brain.schemas.event_schema import Event
from witness_brain.schemas.brain_schema import BrainOutput

# MLX imports (lazy loaded)
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

class BrainClient:
    """Client for the Brain layer - calls LLM to generate thoughts and responses."""

    def __init__(self, config_path="witness_brain/config/brain.yaml"):
        """
        Initialize the BrainClient with configuration.

        Args:
            config_path: Path to brain.yaml config file
        """
        self.config = self._load_config(config_path)

        # Model settings
        model_config = self.config.get("model", {})
        self.model_type = model_config.get("type", "stub")
        self.endpoint = model_config.get("endpoint", "http://localhost:11434/api/chat")
        self.model_name = model_config.get("model_name", "llama3.2:3b-instruct-q4_K_M")

        # Behavior settings
        behavior_config = self.config.get("behavior", {})
        self.max_events = behavior_config.get("max_events", 10)
        self.system_prompt = behavior_config.get("system_prompt", "You are Witness.")

        # MLX model (lazy loaded)
        self._mlx_model = None
        self._mlx_tokenizer = None

        logger.info(f"BrainClient initialized: type={self.model_type}, model={self.model_name}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Brain config not found at {config_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing brain config: {e}")
            return {}

    def _format_events(self, events: List[Event]) -> str:
        """
        Format events into a readable string for the LLM prompt.

        Args:
            events: List of Event objects

        Returns:
            Formatted string representation of events
        """
        if not events:
            return "No recent events."

        lines = []
        for event in events:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            lines.append(f"- [{timestamp}] [{event.source.upper()}] {event.content}")

        return "\n".join(lines)

    def _get_stub_response(self, events: List[Event]) -> BrainOutput:
        """
        Return a stubbed response for testing without LLM.

        Args:
            events: List of recent events

        Returns:
            Stubbed BrainOutput
        """
        # Get the most recent hearing event for context
        last_utterance = "something"
        for event in reversed(events):
            if "hearing" in event.source:
                last_utterance = event.content[:50]
                break

        return BrainOutput(
            thought=f"I heard the user say: '{last_utterance}'. I should respond thoughtfully.",
            speech=f"I heard you. Let me think about that.",
            action="log_mood_observation"
        )

    def _call_ollama(self, events: List[Event]) -> BrainOutput:
        """
        Call Ollama API to generate a response.

        Args:
            events: List of recent events

        Returns:
            BrainOutput from LLM response
        """
        formatted_events = self._format_events(events)

        user_message = f"""Here are the recent events:

{formatted_events}

Based on these events, respond with a JSON object containing your thought, speech, and action."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "{}")

            # Parse JSON response
            try:
                data = json.loads(content)
                return BrainOutput(
                    thought=data.get("thought", "No thought generated."),
                    speech=data.get("speech", "I'm not sure what to say."),
                    action=data.get("action")
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                return self._get_stub_response(events)

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return self._get_stub_response(events)
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {self.endpoint}")
            return self._get_stub_response(events)
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return self._get_stub_response(events)

    def _ensure_mlx_model(self):
        """Lazy load MLX model on first use."""
        if self._mlx_model is not None:
            return

        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available. Install with: pip install mlx-lm")

        logger.info(f"Loading MLX model: {self.model_name}")
        self._mlx_model, self._mlx_tokenizer = mlx_load(self.model_name)
        logger.info("MLX model loaded")

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from model output text."""
        # Try to find JSON in the text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            return match.group()
        return text

    def _generate_mlx(self, events: List[Event]) -> BrainOutput:
        """
        Generate response using local MLX model.

        Args:
            events: List of recent events

        Returns:
            BrainOutput from MLX model
        """
        self._ensure_mlx_model()

        events_text = self._format_events(events)

        # Simple prompt for fast generation
        prompt = f"""You are Witness, a calm companion. Given these events, respond with JSON containing thought, speech, action.

Events:
{events_text}

Respond ONLY with valid JSON like: {{"thought": "...", "speech": "...", "action": null}}
JSON:"""

        try:
            result_text = mlx_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=prompt,
                max_tokens=150,
                verbose=False
            )

            # Parse JSON from response
            json_str = self._extract_json(result_text)
            data = json.loads(json_str)

            return BrainOutput(
                thought=data.get("thought", "").strip(),
                speech=data.get("speech", "I'm here.").strip(),
                action=data.get("action")
            )

        except json.JSONDecodeError as e:
            logger.error(f"MLX JSON parse error: {e}")
            logger.debug(f"Raw MLX output: {result_text}")
            return BrainOutput(
                thought="Had trouble parsing my thoughts.",
                speech="I'm here with you.",
                action=None
            )
        except Exception as e:
            logger.error(f"MLX generation error: {e}")
            return self._get_stub_response(events)

    def generate(self, events: List[Event]) -> BrainOutput:
        """
        Generate a BrainOutput from recent events.

        Args:
            events: List of Event objects

        Returns:
            BrainOutput with thought, speech, and optional action
        """
        # Truncate to max_events
        recent_events = events[-self.max_events:] if len(events) > self.max_events else events

        if self.model_type == "stub":
            logger.info("Using stub brain response")
            return self._get_stub_response(recent_events)
        elif self.model_type == "ollama":
            logger.info(f"Calling Ollama model: {self.model_name}")
            return self._call_ollama(recent_events)
        elif self.model_type == "mlx_local":
            logger.info(f"Calling MLX model: {self.model_name}")
            return self._generate_mlx(recent_events)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using stub")
            return self._get_stub_response(recent_events)


# Test the brain client
if __name__ == "__main__":
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    # Create test events
    test_events = [
        Event(source="hearing:wake", content="hey witness"),
        Event(source="hearing:command", content="how are you today"),
    ]

    client = BrainClient()
    output = client.generate(test_events)

    print(f"\nBrain Output:")
    print(f"  Thought: {output.thought}")
    print(f"  Speech: {output.speech}")
    print(f"  Action: {output.action}")
