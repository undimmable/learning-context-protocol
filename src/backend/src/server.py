from flask import Flask, request, jsonify, send_from_directory

app = Flask("LCP")


# Flask routes
@app.route('/')
def index():
    return send_from_directory('../../..', 'index.html')


@app.route('/webgazer.js')
def webgazer_js():
    return send_from_directory('../../..', 'webgazer.js')

@app.route('/collect_gaze_data', methods=['POST'])
def collect_gaze_data():
    if not request.json or 'gazeData' not in request.json:
        return jsonify({"error": "Invalid request"}), 400

    # Store gaze data in buffer
    global gaze_data_buffer
    gaze_data_buffer.extend(request.json['gazeData'])

    # Process data when we have enough samples
    if len(gaze_data_buffer) >= 100:  # Process after collecting 100 points
        processed_data = process_gaze_data(gaze_data_buffer)

        # Log the processed data
        log_entry({
            "type": "gaze_data",
            "processed": processed_data
        })

        # Fine-tune model with this batch
        fine_tune_result = fine_tune_model_with_gaze_mood_data([processed_data])

        # Clear buffer
        gaze_data_buffer = []

        return jsonify({
            "status": "success",
            "message": "Data processed and model updated",
            "fine_tune_result": fine_tune_result,
            "inferred_mood": processed_data["mood"]
        })

    return jsonify({"status": "success", "message": f"Collected {len(request.json['gazeData'])} gaze points"})


@app.route('/update_context', methods=['POST'])
def update_context():
    """
    Allow users to directly update their context/mood
    """
    if not request.json:
        return jsonify({"error": "Invalid request"}), 400

    global context_data

    # Update context with user-provided information
    if 'mood' in request.json:
        context_data["current_mood"] = request.json['mood']
        context_data["mood_history"].append({
            "mood": request.json['mood'],
            "timestamp": timestamp_string(),
            "source": "user_reported"
        })
        context_data["mood_history"] = context_data["mood_history"][-10:]

    if 'activity' in request.json:
        context_data["recent_activities"].append(request.json['activity'])
        context_data["recent_activities"] = context_data["recent_activities"][-5:]

    if 'focus_level' in request.json:
        context_data["focus_level"] = float(request.json['focus_level'])

    if 'fatigue_level' in request.json:
        context_data["fatigue_level"] = float(request.json['fatigue_level'])

    log_entry({
        "type": "context_update",
        "source": "user_input",
        "new_context": context_data
    })

    return jsonify({
        "status": "success",
        "message": "Context updated",
        "current_context": context_data
    })

@app.route('/provide_feedback', methods=['POST'])
def provide_feedback():
    """
    Allow users to provide feedback on model responses to improve learning
    """
    if not request.json or 'exchange_id' not in request.json or 'rating' not in request.json:
        return jsonify({"error": "Invalid request"}), 400

    feedback = {
        "exchange_id": request.json['exchange_id'],
        "rating": request.json['rating'],
        "corrected_response": request.json.get('corrected_response'),
        "feedback_text": request.json.get('feedback_text'),
        "timestamp": timestamp_string()
    }

    # Store the feedback
    with open("user_feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback) + "\n")

    # If this was a good exchange, store it for future training
    if feedback["rating"] >= 4 and 'original_prompt' in request.json and 'original_response' in request.json:
        with open("successful_exchanges.jsonl", "a") as f:
            f.write(json.dumps({
                "prompt": request.json['original_prompt'],
                "response": request.json['original_response'],
                "feedback_score": feedback["rating"],
                "timestamp": timestamp_string()
            }) + "\n")

    # If user provided corrected response, use it for immediate learning
    if feedback["corrected_response"] and 'original_prompt' in request.json:
        fine_tune_result = fine_tune_model_with_gaze_mood_data(
            [{"intent": "corrected example", "certainty": 1.0, "urgency": 0.5}],
            feedback={
                "corrected_response": feedback["corrected_response"]
            }
        )
        return jsonify({
            "status": "success",
            "message": "Feedback recorded and model updated",
            "fine_tune_result": fine_tune_result
        })

    return jsonify({"status": "success", "message": "Feedback recorded"})

@app.route('/run_agent', methods=['POST'])
def api_run_agent():
    if not request.json:
        return jsonify({"error": "Invalid request"}), 400

    intent = request.json.get('intent', '')
    certainty = float(request.json.get('certainty', 0.5))
    urgency = float(request.json.get('urgency', 0.5))
    include_mood = request.json.get('include_mood', True)

    result = run_agent(intent, certainty, urgency, include_mood)

    # Generate a unique ID for this exchange
    exchange_id = f"ex_{timestamp_string()}_{hash(intent) % 10000}"

    return jsonify({
        "exchange_id": exchange_id,
        "result": result,
        "context": context_data if include_mood else None
    })

@app.route('/set_window_size', methods=['POST'])
def set_window_size():
    if not request.json or 'width' not in request.json or 'height' not in request.json:
        return jsonify({"error": "Invalid request"}), 400

    global window_width, window_height
    window_width = request.json['width']
    window_height = request.json['height']

    return jsonify({"status": "success", "message": "Window size updated"})