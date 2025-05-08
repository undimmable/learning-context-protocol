# Learning Context Protocol (MCP v2)

## Overview

Learning Context Protocol (MCP v2) is a protocol implementation for advanced learning based on empathetic signals processing. It uses gaze tracking and natural language processing to understand and respond to users' emotional states and needs.

## Features

- Real-time gaze tracking and analysis
- Multi-modal context understanding (gaze patterns, text input, user feedback)
- Emotional state inference using FLAN-T5 model
- Interactive web interface for real-time feedback
- Continuous learning through user interactions
- Fine-tuning capabilities for improved emotional intelligence 

## Tech Stack

- Backend: Python (Flask)
- Frontend: HTML/JavaScript (WebGazer.js)
- ML Model: FLAN-T5
- Data Storage: JSONL files

## Current State

The system currently:
- Tracks and analyzes user gaze patterns
- Infers emotional states from multiple data sources
- Provides contextual responses based on intent and mood
- Learns from user feedback
- Supports real-time interaction through web interface

## Goals

1. Improve emotional intelligence in human-computer interactions
2. Create more natural and empathetic AI responses
3. Build a comprehensive understanding of user states through multi-modal analysis
4. Enable continuous learning from user interactions

## Next Steps

1. Enhanced Model Training
   - Collect more diverse emotional training data
   - Implement advanced feature extraction from gaze patterns
   - Add support for more emotional indicators

2. System Improvements
   - Add error handling and recovery mechanisms
   - Implement data validation and sanitization
   - Improve response generation quality

3. User Experience
   - Add more detailed mood tracking
   - Improve the feedback mechanism
   - Enhance the web interface

## Installation

```bash
pip install -r requirements.txt
```
## Usage

1. Start the Flask server:
```bash
python src/backend/main.py serve
```

2. Open the web interface in your browser:
http://localhost:5000

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.