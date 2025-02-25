# Open Source LLM Examples

This directory contains example implementations of conversation detection using open-source LLMs. These examples serve as starting points for implementing your own models or using different open-source alternatives.

## Part 1: Ollama Setup & Configuration

<details>
<summary><strong>Setting up Ollama for Self-Hosted Models</strong></summary>

Ollama is a framework that allows you to run open-source language models locally. It provides:
- Easy model management and deployment
- Local inference without external API calls
- Support for multiple open-source models
- Simple REST API interface

### Installation

1. Install Ollama:
   ```bash
   # macOS or Linux
   curl https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. Start Ollama Server:
   ```bash
   ollama serve
   ```

### Server Configuration

- Default port: 11434
- API endpoints:
  * `/api/chat`: For chat-style interactions
  * `/api/generate`: For completion-style interactions
- Environment variables:
  * `OLLAMA_PORT`: Custom port (optional)

### Managing Models

```bash
# List available models
ollama list

# Pull a model
ollama pull MODEL_NAME

# Remove a model
ollama rm MODEL_NAME
```

### Troubleshooting

<details>
<summary>"Ollama server is not running"</summary>

- Make sure you've started Ollama with `ollama serve`
- Check if Ollama is running on the default port (11434)
- If using a custom port, specify it with `--port PORT_NUMBER`
</details>

<details>
<summary>"Error: listen tcp 127.0.0.1:11434: bind: address already in use"</summary>

- This means Ollama is already running on your machine
- No need to start it again - you can proceed with using the model
- If you need to restart Ollama:
  1. Find the process: `ps aux | grep ollama`
  2. Stop it: `killall ollama`
  3. Start again: `ollama serve`
</details>

</details>

## Part 2: Adding a Self-Hosted, Open-Source Model

<details>
<summary><strong>Example Implementation: Mistral 7B</strong></summary>

This example demonstrates how to implement conversation detection using the Mistral 7B model, a powerful open-source LLM.

### Model Overview
- Name: Mistral 7B
- License: Apache 2.0
- Size: ~4GB
- Requirements:
  * 8GB RAM minimum
  * 4GB free disk space
  * CPU with AVX2 support

### Setup Steps

1. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

2. Install Python dependencies:
   ```bash
   pip install requests
   ```

### Usage

Run the implementation on your message data:

```bash
# The script will automatically generate the output filename following the convention:
# labels_[YYYYMMDD_HHMMSS]_mistral7b_[group_name].csv
python mistral.py path/to/your/messages.csv [--port PORT_NUMBER]
```

For example, using the Cere Network community data (run from the root of the repo):
```bash
python llm_benchmarks/open_source_examples/mistral.py frontend/public/data/groups/thisiscere/messages_thisiscere.csv
```

If you want to run from where this README is located (/llm_benchmarks/open_source_examples/):
```bash
python mistral.py ../../frontend/public/data/groups/thisiscere/messages_thisiscere.csv
```   


### Output Format

The script generates a CSV file with the following columns:
- `message_id`: Original message identifier
- `conversation_id`: Assigned conversation group (0 for spam messages)
- `topic`: Descriptive topic for the conversation
- `timestamp`: Original message timestamp
- `labeler_id`: Always "mistral7b"
- `confidence`: Float between 0 and 1 indicating confidence in the classification

Example output:
```csv
message_id,conversation_id,topic,timestamp,labeler_id,confidence
36598,0,Spam Messages,2021-07-14T14:26:50Z,mistral7b,0.99
36635,1,Giveaway,2025-01-15T02:52:44Z,mistral7b,0.95
36638,2,Partnership Inquiry,2025-01-15T04:31:48Z,mistral7b,0.95
```

### Implementation Details

The implementation showcases:
1. Model initialization and connection to Ollama
2. Batch processing for efficient inference
3. Robust error handling
4. Standardized output formatting
5. Confidence scoring

Key features:
- Automatic conversation grouping
- Spam detection (conversation_id = 0)
- Topic generation
- Confidence scoring
- Standard naming conventions

### Troubleshooting Model-Specific Issues

<details>
<summary>Out of memory errors</summary>

- Reduce `BATCH_SIZE` in `mistral.py` (default: 25)
- Close other resource-intensive applications
- Consider using a smaller model variant
</details>

<details>
<summary>Model not responding</summary>

- Check Ollama server status
- Verify model is properly downloaded
- Try restarting Ollama server
</details>

</details>

## Implementation Guide

<details>
<summary><strong>Adding Your Own Model</strong></summary>

To add support for a different open-source model:

1. Ensure Ollama supports your model
2. Pull the model using `ollama pull MODEL_NAME`
3. Create a new Python file based on `mistral.py`
4. Update the following constants:
   ```python
   MODEL_NAME = "your_model_name"  # Name used with Ollama
   MODEL_ID = "your_model_id"      # ID for output files
   BATCH_SIZE = 25                 # Adjust based on model capacity
   ```
5. Implement the required methods:
   - `__init__`: Model initialization
   - `detect`: Conversation detection logic
   - `load_messages`: Message loading
   - `save_labels`: Results saving

Your implementation should:
- Follow the same input/output formats
- Use consistent naming conventions
- Handle errors appropriately
- Include clear documentation
</details>

<details>
<summary><strong>Contributing Guidelines</strong></summary>

Feel free to contribute your own implementations! Just:
1. Follow the existing code structure
2. Include clear setup instructions
3. Document system requirements
4. Add appropriate error handling
5. Submit a pull request
</details>

<details>
<summary><strong>Planned Future Additions</strong></summary>

We're planning to add examples for:
- Llama 2
- Phi-2
- TinyLlama
- Other community suggestions!
</details> 