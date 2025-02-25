import os
import pandas as pd
import json
from anthropic import Anthropic
from typing import Dict, List, Set, Tuple
import glob
from collections import defaultdict
from datetime import datetime

def extract_group_name(path: str) -> str:
    """Extract group name from a path."""
    return os.path.basename(path.rstrip('/'))

def load_messages(group_dir: str) -> pd.DataFrame:
    """Load raw messages for a group."""
    group_name = extract_group_name(group_dir)
    messages_path = os.path.join(group_dir, f"messages_{group_name}.csv")
    return pd.read_csv(messages_path)

def load_labels(group_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all label files for a group, returns dict mapping model name to labels DataFrame."""
    group_name = extract_group_name(group_dir)
    label_files = glob.glob(os.path.join(group_dir, f"labels_*_{group_name}.csv"))
    labels_by_model = {}
    
    for file in label_files:
        # Extract model name from filename
        # Format: labels_YYYYMMDD_MODEL_group.csv
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 3:
            model = parts[2]  # Get the actual model name (e.g., gpt4o, claude35s, deepseekv3)
            try:
                df = pd.read_csv(file)
                labels_by_model[model] = df
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            
    return labels_by_model

def get_unique_topics(labels_df: pd.DataFrame) -> Set[str]:
    """Extract unique topics from a labels DataFrame."""
    return set(labels_df['topic'].unique())

def get_conversation_for_topic(messages_df: pd.DataFrame, labels_df: pd.DataFrame, topic: str) -> str:
    """Get a representative conversation for a topic."""
    # Get message IDs for this topic
    topic_messages = labels_df[labels_df['topic'] == topic]
    conv_id = topic_messages.iloc[0]['conversation_id']
    message_ids = topic_messages[topic_messages['conversation_id'] == conv_id]['message_id'].tolist()
    
    # Get messages for these IDs
    conv_messages = messages_df[messages_df['id'].isin(message_ids)]
    
    # Sort by timestamp to maintain conversation flow
    conv_messages = conv_messages.sort_values('timestamp')
    
    # Format messages into a string
    conversation = "\n".join([
        f"{row['username'] or row['first_name']}: {row['text']}"
        for _, row in conv_messages.iterrows()
    ])
    
    return conversation

def evaluate_topic(client: Anthropic, conversation: str, topic: str) -> Dict[str, float]:
    """Use Claude to evaluate a topic label based on the criteria."""
    with open("llm_benchmarks/evaluate_topic_labels_prompt.txt", "r") as f:
        system_prompt = f.read()
        
    user_prompt = (
        f"Conversation content:\n{conversation}\n\n"
        f"Topic Label: {topic}\n\n"
        "Please evaluate this topic label according to the criteria and provide numerical scores.\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        "{\n"
        '  "information_density": <score 1-10>,\n'
        '  "redundancy": <score 1-10>,\n'
        '  "relevance": <score 1-10>,\n'
        '  "efficiency": <score 1-10>,\n'
        '  "overall": <average of all scores>\n'
        "}\n\n"
        "Do not include any other text in your response, only the JSON object."
    )
    
    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        temperature=0,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": user_prompt
        }]
    )
    
    try:
        # Extract JSON from response
        response_text = message.content[0].text
        print(f"Claude response for '{topic}':\n{response_text}\n")
        scores = json.loads(response_text)
        return scores
    except Exception as e:
        print(f"Error parsing Claude's response for topic '{topic}': {e}")
        print(f"Raw response:\n{response_text}")
        return {
            'information_density': 0,
            'redundancy': 0,
            'relevance': 0,
            'efficiency': 0,
            'overall': 0
        }

def evaluate_group(group_dir: str, anthropic_api_key: str):
    """Evaluate all topics for a group and save results."""
    client = Anthropic(api_key=anthropic_api_key)
    group_name = extract_group_name(group_dir)
    
    print(f"Evaluating topics for group: {group_name}")
    print(f"Loading data from directory: {group_dir}")
    
    # Load data
    messages_df = load_messages(group_dir)
    labels_by_model = load_labels(group_dir)
    
    if not labels_by_model:
        print("No label files found!")
        return
    
    print(f"Found label files for models: {', '.join(labels_by_model.keys())}")
    results = []
    
    for model, labels_df in labels_by_model.items():
        unique_topics = get_unique_topics(labels_df)
        print(f"\nProcessing {len(unique_topics)} unique topics for model {model}")
        model_scores = defaultdict(list)
        
        for i, topic in enumerate(unique_topics, 1):
            print(f"  Evaluating topic {i}/{len(unique_topics)}: {topic}")
            conversation = get_conversation_for_topic(messages_df, labels_df, topic)
            scores = evaluate_topic(client, conversation, topic)
            
            # Store individual topic scores
            results.append({
                'model': model,
                'topic': topic,
                'information_density': scores['information_density'],
                'redundancy': scores['redundancy'],
                'relevance': scores['relevance'],
                'efficiency': scores['efficiency'],
                'overall': scores['overall']
            })
            
            # Accumulate scores for model averages
            for criterion, score in scores.items():
                model_scores[criterion].append(score)
        
        # Calculate and store model averages
        model_averages = {
            'model': model,
            'topic': 'AVERAGE',
            'information_density': sum(model_scores['information_density']) / len(model_scores['information_density']),
            'redundancy': sum(model_scores['redundancy']) / len(model_scores['redundancy']),
            'relevance': sum(model_scores['relevance']) / len(model_scores['relevance']),
            'efficiency': sum(model_scores['efficiency']) / len(model_scores['efficiency']),
            'overall': sum(model_scores['overall']) / len(model_scores['overall'])
        }
        results.append(model_averages)
        print(f"\nModel {model} average scores:")
        for criterion, score in model_averages.items():
            if criterion not in ('model', 'topic'):
                print(f"  {criterion}: {score:.2f}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save results with timestamp
    output_path = os.path.join(group_dir, f"metrics_topics_{group_name}_{timestamp}.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved topic evaluation results to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate_topics.py <group_directory>")
        sys.exit(1)
        
    group_dir = sys.argv[1]
    if not os.path.isdir(group_dir):
        print(f"Error: Directory does not exist: {group_dir}")
        sys.exit(1)
        
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
        
    evaluate_group(group_dir, anthropic_api_key) 