"""Periodic Fine-Tuning for the MCP Server.

This module provides functionality for periodically fine-tuning the GPTLikeModel
used by the MCP Server. It runs the fine-tuning process in a background thread
to avoid affecting the performance of the MCP Server.

Usage:
    from src.tools.periodic_fine_tuning import PeriodicFineTuner

    # Initialize the fine-tuner
    fine_tuner = PeriodicFineTuner(
        interval_hours=24,  # Fine-tune once per day
        data_path="data/mcp_training_data.json",
        model_path="models/gpt_model.pt",
        output_path="models/gpt_model_finetuned.pt"
    )

    # Start the fine-tuning process
    fine_tuner.start()

    # Later, when shutting down the server
    fine_tuner.stop()
"""

import os
import time
import json
import threading
import logging
import datetime
from typing import Optional, Dict, Any, List, Callable

import torch

from src.neural.training import fine_tune_on_mcp_data
from src.db.memory_store import MemoryStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("periodic_fine_tuning")

class PeriodicFineTuner:
    """Class for periodically fine-tuning the GPTLikeModel."""

    def __init__(
        self,
        interval_hours: float = 24.0,
        data_path: str = "data/mcp_training_data.json",
        model_path: Optional[str] = None,
        output_path: str = "models/gpt_model_finetuned.pt",
        memory_store: Optional[MemoryStore] = None,
        on_model_updated: Optional[Callable[[str], None]] = None,
        vocab_size: int = 30522,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        depth: int = 6,
        max_seq_len: int = 512,
        use_attention: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.01,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        epochs: int = 3,
        tokenizer_name: str = "bert-base-uncased"
    ):
        """Initialize the PeriodicFineTuner.

        Args:
            interval_hours: Interval between fine-tuning runs in hours.
            data_path: Path to save the training data.
            model_path: Path to the current model.
            output_path: Path to save the fine-tuned model.
            memory_store: MemoryStore instance to get training data from.
            on_model_updated: Callback function to call when the model is updated.
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the embeddings.
            hidden_dim: Dimension of the hidden layers.
            depth: Number of layers.
            max_seq_len: Maximum sequence length.
            use_attention: Whether to use attention mechanisms.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            alpha: Alpha parameter for LeakyReLU.
            batch_size: Batch size for training.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            tokenizer_name: Name of the HuggingFace tokenizer to use.
        """
        self.interval_seconds = interval_hours * 3600
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.memory_store = memory_store
        self.on_model_updated = on_model_updated
        
        # Fine-tuning parameters
        self.fine_tuning_params = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "max_seq_len": max_seq_len,
            "use_attention": use_attention,
            "num_heads": num_heads,
            "dropout": dropout,
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "tokenizer_name": tokenizer_name
        }
        
        # Thread control
        self.stop_event = threading.Event()
        self.thread = None
        self.last_fine_tune_time = None
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _prepare_training_data(self) -> bool:
        """Prepare training data from memory store.
        
        Returns:
            True if data was successfully prepared, False otherwise.
        """
        if not self.memory_store:
            logger.warning("No memory store provided, cannot prepare training data")
            return False
            
        try:
            # Get all entries from memory store
            entries = self.memory_store.all()
            
            if not entries:
                logger.warning("No entries found in memory store")
                return False
                
            # Format entries for training
            training_data = []
            for entry in entries:
                # Extract text and format as a training example
                text = entry.get("text", "")
                if text:
                    # Simple format: each memory entry becomes a training example
                    training_data.append({
                        "prompt": "Remember this information:",
                        "response": text
                    })
            
            if not training_data:
                logger.warning("No valid training data could be extracted")
                return False
                
            # Save training data to file
            with open(self.data_path, 'w') as f:
                json.dump(training_data, f)
                
            logger.info(f"Prepared {len(training_data)} training examples")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return False

    def _fine_tune(self):
        """Run the fine-tuning process."""
        try:
            logger.info("Starting fine-tuning process")
            
            # Prepare training data
            if not self._prepare_training_data():
                logger.warning("Failed to prepare training data, skipping fine-tuning")
                return
                
            # Determine model path
            current_model_path = self.model_path
            if not current_model_path or not os.path.exists(current_model_path):
                if os.path.exists(self.output_path):
                    # Use the previously fine-tuned model
                    current_model_path = self.output_path
                    logger.info(f"Using previously fine-tuned model: {current_model_path}")
                else:
                    # No model available, will start from scratch
                    current_model_path = None
                    logger.info("No existing model found, will initialize a new one")
            
            # Run fine-tuning
            fine_tune_on_mcp_data(
                model_path=current_model_path,
                data_path=self.data_path,
                output_path=self.output_path,
                **self.fine_tuning_params
            )
            
            # Update last fine-tune time
            self.last_fine_tune_time = datetime.datetime.now()
            
            # Call the callback if provided
            if self.on_model_updated and os.path.exists(self.output_path):
                self.on_model_updated(self.output_path)
                
            logger.info("Fine-tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")

    def _fine_tuning_loop(self):
        """Background thread function for periodic fine-tuning."""
        logger.info(f"Fine-tuning thread started, interval: {self.interval_seconds} seconds")
        
        while not self.stop_event.is_set():
            # Run fine-tuning
            self._fine_tune()
            
            # Wait for the next interval or until stopped
            logger.info(f"Next fine-tuning scheduled in {self.interval_seconds} seconds")
            self.stop_event.wait(self.interval_seconds)
            
        logger.info("Fine-tuning thread stopped")

    def start(self):
        """Start the periodic fine-tuning process."""
        if self.thread and self.thread.is_alive():
            logger.warning("Fine-tuning thread is already running")
            return
            
        logger.info("Starting periodic fine-tuning")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._fine_tuning_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the periodic fine-tuning process."""
        if not self.thread or not self.thread.is_alive():
            logger.warning("Fine-tuning thread is not running")
            return
            
        logger.info("Stopping periodic fine-tuning")
        self.stop_event.set()
        self.thread.join(timeout=10)  # Wait up to 10 seconds for the thread to stop
        
        if self.thread.is_alive():
            logger.warning("Fine-tuning thread did not stop gracefully")
        else:
            logger.info("Fine-tuning thread stopped successfully")
            self.thread = None

    def fine_tune_now(self):
        """Trigger fine-tuning immediately."""
        logger.info("Manual fine-tuning triggered")
        
        # Run in a separate thread to avoid blocking
        thread = threading.Thread(target=self._fine_tune, daemon=True)
        thread.start()
        return thread

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the fine-tuning process.
        
        Returns:
            A dictionary with status information.
        """
        return {
            "running": self.thread is not None and self.thread.is_alive(),
            "last_fine_tune_time": self.last_fine_tune_time.isoformat() if self.last_fine_tune_time else None,
            "interval_hours": self.interval_seconds / 3600,
            "model_path": self.model_path,
            "output_path": self.output_path,
            "data_path": self.data_path
        }