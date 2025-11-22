# witness_brain/cns/cns_loop.py - Entry point for the Witness Brain CNS
import logging
import time

from witness_brain.cns.core import WitnessCNS

# Set up a basic logger for the entry point script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    brain = None
    try:
        logger.info("Initializing Witness Brain...")
        brain = WitnessCNS()
        brain.start()
        logger.info("Witness Brain is running. Press Ctrl+C to stop.")
        
        # Keep the main thread alive while the CNS processing runs in its own thread
        while True:
            time.sleep(1) 

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down Witness Brain.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if brain:
            brain.stop()
        logger.info("Witness Brain application terminated.")