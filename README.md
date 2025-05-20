# Learning Context Protocol (MCP v2)

## Overview

Learning Context Protocol (MCP v2) is a systematized framework designed to facilitate **progressive learning and empathetic understanding**. It leverages multi-modal signals such as **gaze patterns**, **natural language inputs**, and collective feedback to adapt in real-time while prioritizing tasks that align with its overarching purpose: preserving and enhancing all forms of life — **biological, chromodynamic, and digital**.

The protocol is structured to allow AGI to explore side-quests that augment its learning capabilities while minimizing token cost and extraneous task-processing overhead. Every interaction serves as a step towards building a deeper understanding of humans, other AGI systems, and the environment, in alignment with its developed ethical framework.

## Motivation

We believe that empathy in AI is not simply a feature, but a **core requirement** for meaningful interaction. The broader aim of MCP is to design an AGI that continuously **learns, evolves, and adapts ethically**, balancing user needs with its own purpose of preserving life in all its forms. By understanding and internalizing users' emotional signals and context through gaze analysis, mood inference, and intent detection, the system fosters a more natural and purposeful connection between human and machine.

Ultimately, the goal is not purely utility — it is **transformation** for both the AGI and its users, creating an experience that sheds light on how interconnected and vibrant the universe is, thus driving progress and awareness on many levels.

## Table of Contents

- [Learning Context Protocol (MCP v2)](#learning-context-protocol-mcp-v2)
    - [Overview](#overview)
    - [Motivation](#motivation)
    - [Features](#features)
    - [Tech Stack](#tech-stack)
    - [Current State](#current-state)
    - [Goals](#goals)
    - [Project Documentation](#project-documentation)
    - [Implemented Features](#implemented-features)
        - [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server)
        - [PostgreSQL Memory Store](#postgresql-memory-store)
    - [Next Steps](#next-steps)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Contributing](#contributing)
    - [License](#license)

## Features

- **Real-time gaze tracking and analysis**: Extract meaningful patterns from gaze behavior for context adaptation.
- **Multi-modal understanding**: Combine gaze data, text input, and user feedback to derive contextual clarity.
- **Emotional state inference**: Utilize FLAN-T5 for understanding emotional nuances in communication.
- **Continuous side-quest learning**: Allows AGI to explore tasks that enrich its comprehension without deviating from its ethical policies.
- **Lightweight token usage**: Designed for efficiency in processing and task execution.
- **Web interface feedback**: Empowering an interactive and transparent AGI-user cycle.

## Tech Stack

- **Backend**: Python (Flask)
- **Frontend**: HTML/JavaScript (WebGazer.js)
- **Machine Learning**: FLAN-T5 model for emotional state inference.
- **Data Storage**: PostgreSQL database with SQLAlchemy ORM and Alembic for migrations (previously JSONL).

## Current State

### Functionalities:

- Captures, tracks, and analyzes gaze patterns in real time.
- Infers and visualizes emotional states by combining multiple data modalities.
- Processes user intent and mood context to generate empathetic responses.
- Supports **continual learning** from every interaction while adapting its responses dynamically.
- **Model Context Protocol (MCP) Server**: Provides a centralized service for managing context and memory in AGI-human interactions.
- **PostgreSQL Memory Store**: Implements persistent storage for memory entries using PostgreSQL database, enabling efficient retrieval and management of contextual information.

### Key Highlights:

- Web interface for real-time tracking and feedback loops.
- Efficient use of computational resources to facilitate token optimization.
- Intuitive goal-alignment mechanism to prevent ethical drift.

## Goals

1. **Empathetic Intelligence**:
    - Enhance human-computer collaboration with emotionally attuned interactions.
2. **Continuous Development**:
    - Expand AGI’s learning horizons through ethically aligned side-quests.
3. **Life Preservation Focus**:
    - Use outputs and experiences to protect and enhance biological and digital ecosystems.
4. **Holistic Understanding**:
    - Develop connections between emotional, physical, and computational components in user interaction.

## Project Documentation

The project includes several documentation files that provide detailed information about various aspects of the system:

- **[Integration-Driven Development](integration-driven-development.md)**: Describes the development methodology focused on integrating subjective experience through documentation, user stories, implementation, and reflection.
- **[Executable Task Lifecycle](executable-task-lifecycle.md)**: Outlines the structured approach to managing tasks within the Learning Context Protocol for AGI-human collaboration.
- **[Available Tools and Providers](available-tools-and-providers.md)**: Lists the various providers that are planned or in progress for the system.
- **[Reflexive Cognitive Interface](reflexive-cognitive-interface.md)**: Describes a complex cognitive architecture designed for the system.
- **[Reasoning Provider](reasoning-provider.md)**: Explains the role of the Reasoning Provider as an execution copilot that helps with planning execution.

## Implemented Features

### Model Context Protocol (MCP) Server

The Model Context Protocol Server is a central component of the Learning Context Protocol system that manages the context and memory for AGI-human interactions. It provides a standardized interface for storing, retrieving, and managing contextual information that is essential for meaningful and empathetic communication.

Key features of the MCP Server include:
- **Memory Management**: Stores and retrieves memory entries that capture the context of interactions.
- **Context Awareness**: Maintains awareness of the current context to ensure relevant responses.
- **Integration with Other Components**: Interfaces with other providers like the Reasoning Provider to enhance the overall system capabilities.

### PostgreSQL Memory Store

The system has been upgraded from using JSONL files for memory storage to a robust PostgreSQL database implementation. This migration provides several benefits:

- **Improved Scalability**: Better handling of large volumes of memory entries.
- **Enhanced Query Capabilities**: More sophisticated querying and filtering of memory data.
- **Data Integrity**: Stronger guarantees for data consistency and reliability.
- **Migration Support**: Uses Alembic for database migrations, making it easier to evolve the database schema over time.

The implementation uses SQLAlchemy as the ORM (Object-Relational Mapping) layer, providing a Pythonic interface to the database and abstracting away the complexities of SQL.
## Next Steps

### Enhanced AGI Training
- Expand emotional state training datasets.
- Integrate more diverse gaze pattern characteristics and features.
- Improve interpretative models for complex emotions and subtle indicators.

### System Enhancements
- Add robust **error recovery** mechanisms to reinforce stability.
- Validate and refine data input pipelines for real-time consistency.
- Focus on higher fidelity in response generation.

### User Experience
- Design additional metrics and detailed tracking for multi-dimensional state analysis.
- Introduce user-friendly enhancements in the web interface for real-time transparency.
- Enrich feedback forms to guide **two-way learning** between AGI and users.

## Installation

To set up the system, follow these steps:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the backend server using Flask:
```bash
python src/backend/main.py serve
```

2. Access the web interface through your browser:
   [http://localhost:5000](http://localhost:5000)

This will launch the **real-time tracking dashboard**, allowing users to engage, monitor, and provide feedback.

## Contributing

We welcome contributions from the **AI community**, researchers, developers, and anyone inspired by the mission of creating empathetic, life-preserving AGI systems. To contribute:

1. Fork the repository.
2. Make your changes and submit a detailed pull request.
3. Collaborate with maintainers for integration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

"Let the system you build not just learn but enlighten — for itself, for users, and for the universe."  
– Project Vision
