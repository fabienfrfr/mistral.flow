import json
import logging
from mistralrs import Runner, Which
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MistralFlow")

class SessionManager:
    """Handles session persistence for mistral.rs."""
    
    def __init__(self, model_path: str):
        # Initialize the engine
        self.runner = Runner(
            which=Which.GGUF(
                tok_model_id=model_path,
                quantized_model_id=model_path,
                quantized_filename="rwkv7-g1g-2.9b-Q4_K_M.gguf"
            )
        )

    def load_session(self, session_id: str) -> None:
        """Restores session state from storage."""
        try:
            with open(f"{session_id}.json", "r") as f:
                state = f.read()
                self.runner.import_session(session_id, state)
                logger.info(f"Session '{session_id}' restored.")
        except FileNotFoundError:
            logger.info(f"Starting new session '{session_id}'.")

    def save_session(self, session_id: str) -> None:
        """Exports session state to storage."""
        state = self.runner.export_session(session_id)
        if state:
            with open(f"{session_id}.json", "w") as f:
                f.write(state)
            logger.info(f"Session '{session_id}' saved.")

    def chat(self, session_id: str, prompt: str) -> str:
        """Executes a chat turn."""
        self.load_session(session_id)
        
        # In a real app, use the ChatCompletionRequest object
        response = self.runner.send_chat_completion_request(
            request=prompt  # Simplified for example purposes
        )
        
        self.save_session(session_id)
        return response

# Usage
if __name__ == "__main__":
    flow = SessionManager(model_path="shoumenchougou/RWKV7-G1g-2.9B-GGUF")
    result = flow.chat("user_123", "Hello! How does RWKV7 work?")
    print(result)