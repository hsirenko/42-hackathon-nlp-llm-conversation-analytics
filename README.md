# Conversation Analysis Tests with LLMs - Test Suite

## Table of Contents
- [Step 1: Import the evaluation data](#step-1-import-the-evaluation-data)
- [Step 2: Run the evaluation](#step-2-run-the-evaluation)
- [Step 3: Show the results](#step-3-show-the-results)
- [Step 4: Your First Challenge - Experiment with Different Models](#step-4-your-first-challenge-experiment-with-different-models)
- [Step 5: Start understanding the modifying the prompts, pre-grouping techniques, models](#step-5-start-understanding-the-modifying-the-prompts-pre-grouping-techniques-models)

# Step 1: Import the evaluation data

The evaluation data you'll be working with has already been pre-packaged for you so you do not need to worry about parsing or downloading the data.
If you want to look broadly at all the different communities and the files associated with them, you can do so by navigating to the `data/groups` folder. As you'll see, there's 10 different communities associated with different Telegram groups for you to choose from. For our case, we'll be focusing on the Cere Network community, associated with the `thisiscere` Telegram group.

In every community, the data is organized into the following file types:

<details>
<summary><strong>Community Group Chat Data containing the conversations - Inputs to the experimentation</strong></summary>

The input data contains complete conversation histories from each Telegram community group chat, including message content, timestamps, and user information. This serves as the primary source for all our analysis tasks.

  * File path: `data/groups/thisiscere/messages_thisiscere.csv`
  * Contains original conversation content, timestamps, and user information
  * Primary input for all evaluations

    | ID | Text | Timestamp | Username | First Name | Last Name |
    |----|------|-----------|----------|------------|-----------|
    | 36569 | "You create your own attack and burn yourself…it makes no sense when the supply is still 10% and there is no real use case for the $cere token." | 2025-01-14T01:22:56Z | goldgold888 | TT | |
    | 36570 | "That will be improved in the future. I think Burning the supply using tokens from the Treasury is a positive thing. The aim is to reduce inflation." | 2025-01-14T01:25:40Z | Richnd | Richnd | \| I will never DM you first |
    | 36587 | "there was an actual announcement scheduled for today right?" | 2025-01-14T09:40:36Z | jjpdijkstra | Hans | Dijkstra |
    | 36588 | "I for one dont want CERE to miss out on face melting alt season that is not a day longer than q1 of this year." | 2025-01-14T09:42:06Z | jjpdijkstra | Hans | Dijkstra |
    | 36582 | "Confirm Bull run 🎉" | 2025-01-14T09:21:02Z | karwanxoshnaw_marshall | KARWAN | 馬修 克斯 |
</details>

<details>
<summary><strong>Ground Truth Files - Human-verified, manually labeled data that serves as the benchmark for measuring model accuracy</strong></summary>

These files contain human-annotated labels for conversations and spam messages, serving as the gold standard against which we evaluate model performance. Each file represents a different aspect of the ground truth: conversation groupings and spam identification.

  * `data/groups/thisiscere/GT_conversations_thisiscere.csv`: Manual conversation grouping labels

    | Message ID | Conversation ID |
    |------------|----------------|
    | 36569 | 1 |
    | 36570 | 1 |
    | 36587 | 3 |
    | 36588 | 3 |
    | 36582 | 2 |

  * `data/groups/thisiscere/GT_spam_thisiscere.csv`: Manual spam classification labels

    | Message ID | Is Spam |
    |------------|---------|
    | 36569 | 0 |
    | 36570 | 0 |
    | 36587 | 0 |
    | 36588 | 0 |
    | 36582 | 0 |
</details>

For readability's sake, for every individual file, we've provided just a small sample of the data. You're welcome to look at the full data in the `data/groups/thisiscere` folder.

# Step 2: Run the evaluation

## How to run the evaluation - A practical guide

Before running any evaluations, make sure you have the prerequisites and environment set up:

<details>
<summary><strong>A: Environment Setup</strong></summary>

### Prerequisites
- Python 3.8+
- Ollama installed (for running Mistral 7B locally)

### Setup Instructions
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama
# macOS or Windows - Download from https://ollama.ai/download

# Linux
curl https://ollama.ai/install.sh | sh

# Pull the Mistral 7B model
ollama pull mistral
```
</details>

### Main Task: Conversation Clustering

<details>
<summary><strong>Running the Evaluation with Mistral 7B</strong></summary>

We'll use Mistral 7B, a powerful open-source model, to perform conversation clustering. This model provides a good balance between performance and resource requirements.

### Required Files
- Input data: `data/groups/thisiscere/messages_thisiscere.csv`
- Ground truth file: `data/groups/thisiscere/GT_conversations_thisiscere.csv`

### Running the Model
```bash
# Start Ollama server (if not already running)
ollama serve

# Pull the Mistral 7B model (if you haven't already)
ollama pull mistral

# Run the conversation detection
python open_source_examples/mistral.py data/groups/thisiscere/messages_thisiscere.csv
```
</details>

### Additional Tasks

<details>
<summary><strong>C: Running Spam Detection Evaluation (Optional ⭐)</strong></summary>

The spam detection evaluation compares how well different models identify spam messages in a community.

### Required Files
- Ground truth file: `data/groups/thisiscere/GT_spam_thisiscere.csv`
- Model prediction files:
  * GPT-4: `data/groups/thisiscere/labels_20250131_143535_gpt4o_thisiscere.csv`
  * Claude 3.5: `data/groups/thisiscere/labels_20250131_171944_claude35s_thisiscere.csv`
  * DeepSeek V3: `data/groups/thisiscere/labels_20250131_185300_deepseekv3_thisiscere.csv`

### Running the Evaluation
```bash
python spam_metrics.py data/groups/thisiscere
```

### Output
The script will generate:
- Accuracy, precision, recall, and F1 scores for each model
- Results saved as `data/groups/thisiscere/metrics_spam_detection_thisiscere.csv`

Example output:
```csv
model,label_file,accuracy,precision,recall,f1
143535,labels_20250131_143535_gpt4o_thisiscere.csv,1.0,1.0,1.0,1.0      # Perfect spam detection
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.955,0.842,1.0,0.914  # High recall but some false positives
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.939,0.800,1.0,0.889    # Good overall but more false positives
```

Example interpretation from Cere Network results:
```csv
model,label_file,accuracy,precision,recall,f1
143535,labels_20250131_143535_gpt4o_thisiscere.csv,1.0,1.0,1.0,1.0      # Perfect spam detection
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.955,0.842,1.0,0.914  # High recall but some false positives
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.939,0.800,1.0,0.889    # Good overall but more false positives
```

Which translates to this more readable table:

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|---------|-----------|-------|
| GPT-4 | 1.000 | 1.000 | 1.000 | 1.000 | Perfect spam detection |
| DeepSeek V3 | 0.955 | 0.842 | 1.000 | 0.914 | High recall but some false positives |
| Claude 3.5 | 0.939 | 0.800 | 1.000 | 0.889 | Good overall but more false positives |

This table shows that while all models achieve perfect recall (catching all spam), GPT-4 stands out with perfect precision, while DeepSeek V3 and Claude 3.5 occasionally flag legitimate messages as spam.
</details>

<details>
<summary><strong>D: Running Topic Labeling Evaluation (Optional ⭐)</strong></summary>

The topic labeling evaluation assesses the quality and informativeness of conversation topic labels assigned by each model.

### Required Files
- Model prediction files:
  * GPT-4: `data/groups/thisiscere/labels_20250131_143535_gpt4o_thisiscere.csv`
  * Claude 3.5: `data/groups/thisiscere/labels_20250131_171944_claude35s_thisiscere.csv`
  * DeepSeek V3: `data/groups/thisiscere/labels_20250131_185300_deepseekv3_thisiscere.csv`
- Original message content: `data/groups/thisiscere/messages_thisiscere.csv`

### Running the Evaluation
```bash
python evaluate_topics.py data/groups/thisiscere
```

### Output
The script will generate:
- Information density scores
- Redundancy metrics
- Contextual relevance scores
- Label efficiency ratings
- Results saved as `data/groups/thisiscere/metrics_topics_thisiscere.csv`

Example output:
```csv
model,label_file,info_density,redundancy,relevance,efficiency,overall_score
143535,labels_20250131_143535_gpt4o_thisiscere.csv,8.5,0.95,0.92,0.88,0.91      # Excellent topic labeling
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,7.8,0.88,0.85,0.82,0.84  # Good topic labeling
171944,labels_20250131_171944_claude35s_thisiscere.csv,8.2,0.90,0.88,0.85,0.88    # Very good topic labeling
```

Example interpretation from Cere Network results:
```csv
model,label_file,info_density,redundancy,relevance,efficiency,overall_score
143535,labels_20250131_143535_gpt4o_thisiscere.csv,8.5,0.95,0.92,0.88,0.91      # Excellent topic labeling
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,7.8,0.88,0.85,0.82,0.84  # Good topic labeling
171944,labels_20250131_171944_claude35s_thisiscere.csv,8.2,0.90,0.88,0.85,0.88    # Very good topic labeling
```

Which translates to this more readable table:

| Model | Info Density (1-10) | Redundancy (0-1) | Relevance (0-1) | Efficiency (0-1) | Overall Score | Notes |
|-------|-------------------|-----------------|----------------|-----------------|---------------|-------|
| GPT-4 | 8.5 | 0.95 | 0.92 | 0.88 | 0.91 | Excellent topic labeling |
| DeepSeek V3 | 7.8 | 0.88 | 0.85 | 0.82 | 0.84 | Good topic labeling |
| Claude 3.5 | 8.2 | 0.90 | 0.88 | 0.85 | 0.88 | Very good topic labeling |

This table shows that all models perform well at topic labeling, with GPT-4 achieving the highest scores across all metrics. GPT-4 particularly excels in information density and redundancy reduction, while Claude 3.5 maintains strong performance across all categories. DeepSeek V3 shows good results but has slightly lower scores in information density and efficiency.
</details>

<br/>

# Step 3: Show the results

### Main Task

<details>
<summary><strong>Conversation Clustering Results</strong></summary>

<br/>

Results will be stored as `metrics_conversations_[community_name].csv` in your community's folder.

For example, for the Cere Network community: `data/groups/thisiscere/metrics_conversations_thisiscere.csv`

### Model Output Format
The model generates a CSV file following the naming convention:
`data/groups/<community_name>/labels_<YYYYMMDD>_<HHMMSS>_mistral7b_<community_name>.csv`

For example, if you're evaluating the Cere Network community (thisiscere) and run the script on February 25, 2025 at 20:22:30, the output file will be:
`data/groups/thisiscere/labels_20250225_202230_mistral7b_thisiscere.csv`

The file will have the following format (and for readability, we've only included a sample of the data):
```csv
message_id,conversation_id,topic,timestamp,labeler_id,confidence
36598,1,Token Discussion,2021-07-14T14:26:50Z,mistral7b,0.85
36635,2,Project Updates,2025-01-15T02:52:44Z,mistral7b,0.82
36638,0,Spam Message,2025-01-15T04:31:48Z,mistral7b,0.95
```

Which translates to this more readable table:

| Message ID | Conversation ID | Topic | Timestamp | Labeler ID | Confidence |
|------------|----------------|--------|-----------|------------|------------|
| 36598 | 1 | Token Discussion | 2021-07-14T14:26:50Z | mistral7b | 0.85 |
| 36635 | 2 | Project Updates | 2025-01-15T02:52:44Z | mistral7b | 0.82 |
| 36638 | 0 | Spam Message | 2025-01-15T04:31:48Z | mistral7b | 0.95 |

### Performance Metrics

| Model | ARI Score (-1 to 1) | Messages Processed | Notes |
|-------|-------------------|-------------------|-------|
| GPT-4 | 0.583 | 49 | Moderate conversation grouping accuracy |
| Mistral 7B | 0.865 | 67 | Strong conversation grouping, processed more messages |
| Claude 3.5 | 0.568 | 49 | Moderate conversation grouping accuracy |


To visualize or further analyze these results, you can:
1. Load the CSV files into your preferred data analysis tool (Python pandas, Excel, etc.)
2. Create comparative visualizations of model performance
3. Analyze trends across different evaluation aspects
4. Export metrics for integration with other monitoring tools

The standardized CSV format makes it easy to:
- Compare performance across different models
- Track improvements over time
- Identify specific areas where models excel or need improvement
- Generate custom reports and visualizations

Remember: The examples shown above are from the Cere Network community evaluation. When you run the evaluations on a different community, the file paths and content will reflect that community's name and data.

</details>

### Additional Tasks

<details>
<summary><strong>Spam Detection Results (Optional ⭐)</strong></summary>

<br/>

Results will be stored as `metrics_spam_detection_[community_name].csv` in your community's folder.

For example, for the Cere Network community: `data/groups/thisiscere/metrics_spam_detection_thisiscere.csv`

This file contains:
- Accuracy: Overall correctness of spam classification
- Precision: Proportion of true spam among messages flagged as spam
- Recall: Proportion of actual spam messages that were caught
- F1 Score: Balanced measure between precision and recall

Example interpretation from Cere Network results:
```csv
model,label_file,accuracy,precision,recall,f1
143535,labels_20250131_143535_gpt4o_thisiscere.csv,1.0,1.0,1.0,1.0      # Perfect spam detection
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.955,0.842,1.0,0.914  # High recall but some false positives
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.939,0.800,1.0,0.889    # Good overall but more false positives
```

Which translates to this more readable table:

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|---------|-----------|-------|
| GPT-4 | 1.000 | 1.000 | 1.000 | 1.000 | Perfect spam detection |
| DeepSeek V3 | 0.955 | 0.842 | 1.000 | 0.914 | High recall but some false positives |
| Claude 3.5 | 0.939 | 0.800 | 1.000 | 0.889 | Good overall but more false positives |

This table shows that while all models achieve perfect recall (catching all spam), GPT-4 stands out with perfect precision, while DeepSeek V3 and Claude 3.5 occasionally flag legitimate messages as spam.
</details>

<details>
<summary><strong>Topic Labeling Results (Optional ⭐)</strong></summary>

<br/>

Results will be stored as `metrics_topics_[community_name].csv` in your community's folder.

For example, for the Cere Network community: `data/groups/thisiscere/metrics_topics_thisiscere.csv`

This file contains:
- Information Density: How well topics capture essential information (1-10)
- Redundancy: Measure of information efficiency (0-1)
- Relevance: How well topics match conversation content (0-1)
- Efficiency: Optimal use of words in labels (0-1)
- Overall Score: Combined performance metric (0-1)

Example interpretation from Cere Network results:
```csv
model,label_file,info_density,redundancy,relevance,efficiency,overall_score
143535,labels_20250131_143535_gpt4o_thisiscere.csv,8.5,0.95,0.92,0.88,0.91      # Excellent topic labeling
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,7.8,0.88,0.85,0.82,0.84  # Good topic labeling
171944,labels_20250131_171944_claude35s_thisiscere.csv,8.2,0.90,0.88,0.85,0.88    # Very good topic labeling
```

Which translates to this more readable table:

| Model | Info Density (1-10) | Redundancy (0-1) | Relevance (0-1) | Efficiency (0-1) | Overall Score | Notes |
|-------|-------------------|-----------------|----------------|-----------------|---------------|-------|
| GPT-4 | 8.5 | 0.95 | 0.92 | 0.88 | 0.91 | Excellent topic labeling |
| DeepSeek V3 | 7.8 | 0.88 | 0.85 | 0.82 | 0.84 | Good topic labeling |
| Claude 3.5 | 8.2 | 0.90 | 0.88 | 0.85 | 0.88 | Very good topic labeling |

This table shows that all models perform well at topic labeling, with GPT-4 achieving the highest scores across all metrics. GPT-4 particularly excels in information density and redundancy reduction, while Claude 3.5 maintains strong performance across all categories. DeepSeek V3 shows good results but has slightly lower scores in information density and efficiency.
</details>

### Useful links

<details>
<summary><strong>🖥️ Frontend Visualization - Interactive Results Explorer</strong></summary>
<br/>
For a more interactive experience with a graphical user interface, you can access the results through our web application:
- Frontend Application URL: https://conversation-detection.stage.cere.io/
</details>

<details>
<summary><strong>📚 Knowledge Base - Theory behind evaluation metrics</strong></summary>

### Main Task: Conversation Clustering
The primary focus of our evaluation framework is the accurate clustering of messages into coherent conversations. This is the core challenge that directly impacts the quality of community analytics.

<details>
<summary><strong>Conversation Clustering</strong></summary>

The quality of conversation clustering is evaluated using the Adjusted Rand Index (ARI), a standard metric for comparing clustering results:

### Adjusted Rand Index (ARI)
  * Measures the similarity between two clusterings by considering all pairs of messages and checking whether they are grouped together or separately in both clusterings
  * Ranges from -1 to 1, where:
    - 1 indicates perfect agreement with ground truth
    - 0 indicates random labeling
    - Negative values indicate less agreement than expected by chance
  * Advantages:
    - Accounts for chance groupings
    - Handles different numbers of conversations
    - Independent of conversation labels/IDs

### ARI Calculation Process
   * First, convert to pair-wise relationships:
     - Ground Truth pairs in same conversation:
       * (msg1,msg2), (msg1,msg4), (msg2,msg4)  # Group 1
       * (msg3,msg5)                            # Group 2
     
     - Model Output pairs in same conversation:
       * (msg1,msg2)                           # Group 100
       * (msg3,msg5)                           # Group 101
       * msg4 alone in Group 102

   * ARI Score = 0.4 (moderate agreement) because:
     - Correctly grouped: (msg1,msg2), (msg3,msg5)
     - Incorrectly separated: msg4 from (msg1,msg2)

This example demonstrates how even with different conversation IDs (1,2 vs 100,101,102), ARI effectively measures clustering agreement by comparing pair-wise relationships between messages.
</details>

### Additional Tasks
The following tasks complement the main conversation clustering evaluation, providing additional insights into model capabilities:

<details>
<summary><strong>Topic Labeling (Optional ⭐)</strong></summary>

The evaluation of topic labels focuses on how well they capture and convey the essential information from conversations. Using principles from information theory, each topic label is evaluated against the actual conversation content it represents.

### Evaluation Framework
Topic labels are assessed by an expert system using the following information-theoretic criteria:

1. **Information Density** (1-10 scale):
   * Balance between brevity and informativeness
   * Optimal compression of conversation meaning
   * Example: "BTC Price Analysis Q4 2023" (9/10) vs "Crypto Discussion" (3/10)

2. **Redundancy Elimination**:
   * Penalizes repetitive or unnecessary information
   * Measures information efficiency
   * Example: "Bitcoin BTC Crypto Price" (low score due to redundancy) vs "Bitcoin Price Trends" (high score)

3. **Contextual Relevance**:
   * How well the label captures key conversation elements
   * Alignment with actual message content
   * Example: For a technical discussion about blockchain architecture, "Ethereum Gas Optimization" (high relevance) vs "ETH Discussion" (low relevance)

4. **Label Efficiency**:
   * Ratio of useful information to label length
   * Optimal use of each word/term
   * Example: "DeFi Liquidity Pool Returns" (efficient) vs "Discussion About Various Aspects of Decentralized Finance Liquidity Pools" (inefficient)

### Scoring System
Labels are scored on a 1-10 scale where:
- **1-2**: Severely problematic
  * Too vague or incomprehensible
  * Example: "Crypto stuff"
- **3-4**: Poor information value
  * Too generic or extremely redundant
  * Example: "Bitcoin cryptocurrency digital currency discussion"
- **5-6**: Acceptable but suboptimal
  * Conveys basic meaning but lacks precision
  * Example: "Cryptocurrency trading"
- **7-8**: Good balance
  * Clear, informative, efficient
  * Example: "BTC-ETH Price Correlation Analysis"
- **9-10**: Excellent
  * Optimal information density
  * Highly descriptive yet concise
  * Example: "L2 Rollup Performance Benchmarks Q1 2024"

### Example Evaluation

1. **Sample Conversation**:
   ```   User1: "How's Arbitrum's TPS compared to other L2s?"
   User2: "Currently around 40-50k TPS"
   User3: "Optimism is showing similar numbers"
   User1: "What about transaction costs?"
   User2: "Arb slightly cheaper, around $0.1-0.3 per tx"
   ```

2. **Topic Label Evaluation**:
   ```
   Label: "L2 Scaling: Arbitrum vs Optimism Performance"
   Score: 9/10
   Reasoning:
   - Specifies the exact L2 solutions being compared
   - Indicates the comparison is about performance
   - Captures both TPS and cost aspects
   - Concise yet comprehensive
   ```

3. **Alternative Label Analysis**:
   ```
   "L2 Discussion" - Score: 3/10
   - Too vague, loses critical information
   - Fails to capture comparative aspect
   - Missing specific solutions discussed

   "Detailed Technical Analysis of Layer 2 Blockchain Solutions Including Arbitrum and Optimism Transaction Speed Comparisons" - Score: 4/10
   - Unnecessarily verbose
   - High redundancy
   - Poor information-to-length ratio
   ```
</details>

<details>
<summary><strong>Spam Detection (Optional ⭐)</strong></summary>

Spam classification is evaluated using standard binary classification metrics. In our framework, spam messages are identified by `conversation_id = 0` in model outputs.

### Evaluation Metrics
- **Precision**: Accuracy of spam identification (minimize false positives)
  * Formula: `true_positives / (true_positives + false_positives)`
  * Critical for avoiding misclassification of legitimate messages
  * Example: Precision of 0.95 means 95% of messages labeled as spam are actually spam

- **Recall**: Completeness of spam detection (minimize false negatives)
  * Formula: `true_positives / (true_positives + false_negatives)`
  * Important for catching all spam messages
  * Example: Recall of 0.90 means 90% of all actual spam messages were caught

- **F1 Score**: Balanced measure of precision and recall
  * Formula: `2 * (precision * recall) / (precision + recall)`
  * Single metric for overall spam detection performance
  * Helps balance the trade-off between precision and recall

### Example Evaluation

1. **Sample Messages and Ground Truth**:
   ```csv
   message_id,text,is_spam
   msg1,"Check out crypto profits now!",1
   msg2,"What's the BTC price?",0
   msg3,"FREE BITCOIN click here!!!",1
   msg4,"Around $48k right now",0
   msg5,"Make 1000% gains guaranteed!!",1
   ```

2. **Model Output**:
   ```csv
   message_id,conversation_id,confidence
   msg1,0,0.95        # Correctly identified spam
   msg2,1,0.88        # Correctly identified non-spam
   msg3,0,0.92        # Correctly identified spam
   msg4,1,0.85        # Correctly identified non-spam
   msg5,2,0.70        # Missed spam (false negative)
   ```

3. **Metric Calculation**:
   ```
   True Positives (TP) = 2  (msg1, msg3)
   False Positives (FP) = 0
   True Negatives (TN) = 2  (msg2, msg4)
   False Negatives (FN) = 1  (msg5)

   Precision = TP/(TP+FP) = 2/(2+0) = 1.00
   Recall = TP/(TP+FN) = 2/(2+1) = 0.67
   F1 Score = 2 * (1.00 * 0.67)/(1.00 + 0.67) = 0.80
   ```

4. **Confidence Analysis**:
   * High confidence (>0.90) for clear spam patterns
   * Lower confidence (0.70-0.85) for ambiguous cases
   * Threshold of 0.80 used for spam classification

#### Common Spam Patterns
Models are evaluated on their ability to detect:
- Promotional language and excessive punctuation
- Unrealistic promises and urgency
- Suspicious links and contact information
- Repetitive message patterns
- Cross-posting across conversations

The evaluation emphasizes high precision to avoid disrupting legitimate conversations while maintaining acceptable recall for effective spam control.
</details>
</details>

<br/>

# Step 4: Your First Challenge - Experiment with Different Models

Now that you've run the evaluation with Mistral 7B, it's time to experiment with other models! Ollama provides access to various open-source models that you can try.

<details>
<summary><strong>🔧 Practical Guide: How to Experiment with Different Models</strong></summary>

<br/>

1. First, see what models are available:
```bash
ollama list
```

2. Pull additional models you want to try. Here are some suggestions:
```bash
# Try the latest Llama2 model
ollama pull llama2

# Or experiment with CodeLlama
ollama pull codellama

# Or try Phi-2
ollama pull phi
```

3. Modify the example script to work with your chosen model:
- Copy `open_source_examples/mistral.py` to create your own version
- Update the model name and any model-specific parameters
- Run your evaluation with the new model
- Compare the results with Mistral 7B's performance

### Research Questions to Explore

As you experiment with different models, consider these questions:
- How do different model sizes affect the conversation clustering accuracy?
- What's the trade-off between model size and performance?
- Which models are better at specific aspects (topic labeling, spam detection)?
- Can you find a model that performs better than Mistral 7B for your specific use case?

### Tips for Model Experimentation

- Start with smaller models first to understand the evaluation process
- Keep track of your results in a systematic way
- Consider factors like:
  * Model size vs. performance
  * Processing speed
  * Resource requirements
  * Quality of topic labels
  * Accuracy of conversation grouping

Remember to check the [Ollama model library](https://ollama.ai/library) for the latest available models and their specific capabilities.
</details>

<br/>

# Step 5: Start understanding the modifying the prompts, pre-grouping techniques, models

This is the advanced step where you'll start to understand how to modify the prompts, pre-grouping techniques, and models to improve the performance of the model.
