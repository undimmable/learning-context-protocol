import os
import json
import numpy as np
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.serialization.ai_response_tokenizer import AIResponseTypedTokenizer, TypedAIResponse
from src.util.util import timestamp_string


# Load the model and tokenizer
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Store collected data
gaze_data_buffer = []
context_data = {
    "recent_activities": [],
    "mood_history": [],
    "current_mood": "neutral",
    "focus_level": 0.5,
    "fatigue_level": 0.3,
    "recent_interactions": []
}


# Custom dataset for fine-tuning
class GazeDataset(Dataset):
    def __init__(self, input_texts, output_texts):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.input_texts[idx],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        output_encoding = self.tokenizer(
            self.output_texts[idx],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": output_encoding.input_ids.squeeze()
        }


def log_entry(data: dict):
    data["ts"] = timestamp_string() + "Z"
    with open("learning_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def run_agent(intent: str, certainty=0.5, urgency=0.5, include_mood=True) -> TypedAIResponse:
    global context_data

    mood_context = context_data if include_mood else None
    prompt = format_prompt(intent, certainty, urgency, mood_context)

    ai_response = AIResponseTypedTokenizer.tokenize(pipe(prompt, max_new_tokens=100)[0]["generated_text"])

    # Update context with this interaction
    if ai_response.state:
        context_data["recent_interactions"].append({
            "intent": intent,
            "emotional_state": ai_response.state,
            "timestamp": timestamp_string()
        })
        # Keep only the last 5 interactions
        context_data["recent_interactions"] = context_data["recent_interactions"][-5:]

    log_entry({
        "intent": intent,
        "certainty": certainty,
        "urgency": urgency,
        "mood_context": mood_context,
        **ai_response.as_dict()
    })

    return ai_response


def run_batch_from_file(filepath: str, certainty=0.5, urgency=0.5):
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for intent in lines:
        ai_response = run_agent(intent, certainty, urgency)
        print("\nIntent:", intent)
        print("PLAN:", ai_response.plan)
        print("STATE:", ai_response.state)
        if ai_response.note:
            print("NOTE:", ai_response.note)


def process_gaze_data(gaze_data):
    """
    Process the collected gaze data to extract features related to mood and attention
    """
    # Extract features from gaze data
    x_positions = [point['x'] for point in gaze_data]
    y_positions = [point['y'] for point in gaze_data]
    timestamps = [point['timestamp'] for point in gaze_data]

    # Basic gaze features
    avg_x = np.mean(x_positions)
    avg_y = np.mean(y_positions)
    std_x = np.std(x_positions)
    std_y = np.std(y_positions)

    # Advanced gaze metrics that can indicate mood/mental state
    if len(timestamps) > 1:
        # Calculate fixation duration (longer fixations can indicate focus or fatigue)
        dwell_times = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        avg_dwell_time = np.mean(dwell_times) if dwell_times else 0

        # Calculate saccade velocity (fast movements can indicate alertness or anxiety)
        distances = [np.sqrt((x_positions[i + 1] - x_positions[i]) ** 2 +
                             (y_positions[i + 1] - y_positions[i]) ** 2)
                     for i in range(len(x_positions) - 1)]
        velocities = [distances[i] / dwell_times[i] if dwell_times[i] > 0 else 0
                      for i in range(len(distances))]
        avg_velocity = np.mean(velocities) if velocities else 0

        # Blink rate estimation (if available in gaze data)
        # More sophisticated analysis can be added here
    else:
        avg_dwell_time = 0
        avg_velocity = 0

    # Derive mood indicators from gaze patterns
    focus_indicator = min(1.0, max(0.1, 1.0 - (std_x + std_y) / (window_width + window_height)))

    # Longer fixations + lower velocity often indicate fatigue
    fatigue_indicator = min(1.0, max(0.1, (avg_dwell_time / 1000) * (1.0 - min(1.0, avg_velocity / 500))))

    # Determine interest areas (e.g., top of screen, bottom, center)
    screen_region = "center"
    if avg_y < window_height * 0.3:
        screen_region = "top"
    elif avg_y > window_height * 0.7:
        screen_region = "bottom"

    if avg_x < window_width * 0.3:
        screen_region = "left_" + screen_region
    elif avg_x > window_width * 0.7:
        screen_region = "right_" + screen_region

    # Infer intent based on gaze patterns and screen regions
    intent = f"looking at {screen_region}"
    certainty = min(1.0, max(0.1, 1.0 - (std_x / window_width)))
    urgency = min(1.0, max(0.1, avg_velocity / 500))  # Higher velocity might indicate urgency

    # Update global context with this new information
    global context_data
    context_data["focus_level"] = focus_indicator
    context_data["fatigue_level"] = fatigue_indicator

    # Infer mood from gaze patterns
    # This is a simplistic model - could be enhanced with ML techniques
    if focus_indicator > 0.7 and fatigue_indicator < 0.3:
        mood = "focused"
    elif focus_indicator < 0.3 and fatigue_indicator > 0.7:
        mood = "tired"
    elif avg_velocity > 400:  # High velocity
        mood = "agitated"
    elif std_x < window_width * 0.1 and std_y < window_height * 0.1:  # Very stable gaze
        mood = "concentrated"
    else:
        mood = "neutral"

    # Update mood context
    context_data["current_mood"] = mood
    context_data["mood_history"].append({
        "mood": mood,
        "timestamp": timestamp_string()
    })
    context_data["mood_history"] = context_data["mood_history"][-10:]  # Keep last 10 moods

    return {
        "intent": intent,
        "certainty": certainty,
        "urgency": urgency,
        "mood": mood,
        "focus_level": focus_indicator,
        "fatigue_level": fatigue_indicator,
        "screen_region": screen_region,
        "gaze_features": {
            "avg_x": float(avg_x),
            "avg_y": float(avg_y),
            "std_x": float(std_x),
            "std_y": float(std_y),
            "avg_dwell_time": avg_dwell_time,
            "avg_velocity": avg_velocity
        },
        "raw_gaze_data": gaze_data[:5]  # Include the first few data points as sample
    }


def fine_tune_model_with_gaze_mood_data(processed_data_list, feedback=None):
    """
    Fine-tune the FLAN model with gaze data and mood context
    """
    # Prepare training data
    input_texts = []
    output_texts = []

    # Load previous successful exchanges if available
    if os.path.exists("successful_exchanges.jsonl"):
        with open("successful_exchanges.jsonl", "r") as f:
            for line in f:
                try:
                    exchange = json.loads(line)
                    if exchange.get("feedback_score", 0) >= 4:  # Only use highly rated exchanges
                        input_texts.append(exchange["prompt"])
                        output_texts.append(exchange["response"])
                except:
                    pass

    for data in processed_data_list:
        # Include mood context in the prompt
        mood_ctx = {
            "current_mood": data.get("mood", "neutral"),
            "focus_level": data.get("focus_level", 0.5),
            "fatigue_level": data.get("fatigue_level", 0.5),
            "recent_activities": context_data.get("recent_activities", [])
        }

        prompt = format_prompt(data['intent'], data['certainty'], data['urgency'], mood_ctx)
        input_texts.append(prompt)

        # If there's user feedback on a specific response, use that
        if feedback and feedback.get("corrected_response"):
            output_texts.append(feedback["corrected_response"])
        else:
            # Otherwise, use model's current best response
            raw = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
            output_texts.append(raw)

            # Log this exchange
            with open("model_exchanges.jsonl", "a") as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": raw,
                    "timestamp": timestamp_string()
                }) + "\n")

    # If we don't have enough data, don't try to fine-tune
    if len(input_texts) < 5:
        return {"status": "skipped", "message": "Not enough data for fine-tuning yet"}

    # Split data for training
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
        input_texts, output_texts, test_size=0.1
    )

    # Create datasets
    train_dataset = GazeDataset(train_inputs, train_outputs)
    val_dataset = GazeDataset(val_inputs, val_outputs)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # Set up training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

    # Save the fine-tuned model
    model.save_pretrained("finetuned_flan_gaze_model")
    tokenizer.save_pretrained("finetuned_flan_gaze_model")

    # Update the pipeline with the fine-tuned model
    global pipe
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return {"status": "success", "message": "Model fine-tuned with gaze and mood data"}


# Global variables for the window size
window_width = 1920
window_height = 1080

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            run_batch_from_file("intents.txt")
        elif sys.argv[1] == "serve":
            app.run(debug=True)
    else:
        print("TELEPATHY AGENT v1")
        print("Use 'python main.py serve' to start the web server")
        print("Use 'python main.py batch' to run batch processing")
        print("Or continue with interactive mode:")
        intent = input("Intent: ")
        certainty = float(input("Certainty (0–1): ") or 0.5)
        urgency = float(input("Urgency (0–1): ") or 0.5)
        result = run_agent(intent, certainty, urgency)
        print("\n--- Response ---")
        print("PLAN:", result["plan"])
        print("STATE:", result["state"])
        if result["note"]:
            print("NOTE:", result["note"])
