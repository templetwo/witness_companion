# vision/analyzer.py - Vision analysis using Ollama vision models
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class VisionAnalyzer:
    """Analyzes images using Ollama vision models."""

    def __init__(
        self,
        model_name: str = "llama3.2-vision:latest",
        endpoint: str = "http://localhost:11434/api/chat"
    ):
        """
        Initialize the vision analyzer.

        Args:
            model_name: Ollama vision model to use
            endpoint: Ollama API endpoint
        """
        self.model_name = model_name
        self.endpoint = endpoint
        logger.info(f"VisionAnalyzer initialized: model={model_name}")

    def analyze(
        self,
        image_base64: str,
        prompt: str = "Describe what you see briefly. Focus on people, activities, and notable objects."
    ) -> Optional[str]:
        """
        Analyze an image and return a description.

        Args:
            image_base64: Base64 encoded image
            prompt: What to focus on in the analysis

        Returns:
            Description string, or None if analysis failed
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=60  # Vision models need more time
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "")
            return content.strip() if content else None

        except requests.exceptions.Timeout:
            logger.error("Vision analysis timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {self.endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Vision analysis failed: {e}")
            return None

    def describe_scene(self, image_base64: str) -> Optional[str]:
        """
        Get a brief scene description suitable for Witness context.

        Args:
            image_base64: Base64 encoded image

        Returns:
            Brief description for use in Witness events
        """
        prompt = """Describe this scene in 2-3 sentences. Be specific.
Include: who's there (appearance, expression, clothing), what they're doing, the setting.
If there's a person, describe their face (expression, features like glasses/beard/hair).
Don't start with "In this image" or "I can see". Just describe directly."""

        return self.analyze(image_base64, prompt)


# Test the analyzer
if __name__ == "__main__":
    from witness_brain.vision.camera import Camera
    logging.basicConfig(level=logging.INFO)

    cam = Camera()
    analyzer = VisionAnalyzer()

    # Capture and analyze
    image = cam.capture_base64()
    if image:
        print("Analyzing image...")
        description = analyzer.describe_scene(image)
        if description:
            print(f"\nScene: {description}")
        else:
            print("Analysis failed")
    else:
        print("Failed to capture image")

    cam.release()
