# Conversation Analysis Tests with LLMs - Test Suite

## Table of Contents
- [Overview](#overview)
- [Step 1: Import the evaluation data](#step-1-import-the-evaluation-data)
- [Step 2: Run the evaluation](#step-2-run-the-evaluation)
- [Step 3: Show the results](#step-3-show-the-results)
- [Step 4: 🎯 Your First Challenge](#step-4-hot-swap-different-models)
- [Step 5: Start understanding the modifying the prompts, pre-grouping techniques, models](#step-5-start-understanding-the-modifying-the-prompts-pre-grouping-techniques-models)

# Overview

The Conversation Analysis Tests with LLMs module provides a standardized framework for evaluating and comparing different Large Language Models (LLMs) in the context of conversation detection and topic classification. This module currently supports benchmarking of:

- Claude 3.5 Sonnet (Anthropic)
- GPT-4 (OpenAI)
- DeepSeek V3 (DeepSeek)

The benchmarking system is designed to evaluate models on three primary tasks:
1. Conversation Detection: Identifying coherent conversations within a stream of messages
2. Topic Classification: Assigning accurate and informative topic labels to detected conversations
3. Spam Detection: Identifying spam messages within a conversation

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
- Required API keys:
  * Anthropic API key (for Claude)
  * OpenAI API key (for GPT-4)

### Setup Instructions
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# IMPORTANT: Set up API keys for model access
# These are required to run any benchmarks with state-of-the-art models

# Option 1: Set environment variables directly
export OPENAI_API_KEY="your-openai-key"    
export ANTHROPIC_API_KEY="your-anthropic-key"

# Option 2: Use a .env file (recommended)
# Create a .env file in the llm_benchmarks directory with:
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```
</details>

### Main Task

<details>
<summary><strong>B: Running Conversation Clustering Evaluation</strong></summary>

The conversation clustering evaluation assesses how well different models group related messages into coherent conversations.

### Required Files
- Ground truth file: `data/groups/thisiscere/GT_conversations_thisiscere.csv`
- Model prediction files:
  * GPT-4: `data/groups/thisiscere/labels_20250131_143535_gpt4o_thisiscere.csv`
  * Claude 3.5: `data/groups/thisiscere/labels_20250131_171944_claude35s_thisiscere.csv`
  * DeepSeek V3: `data/groups/thisiscere/labels_20250131_185300_deepseekv3_thisiscere.csv`

### Running the Evaluation
```bash
python conversation_metrics.py data/groups/thisiscere
```

### Output
The script will generate:
- Adjusted Rand Index (ARI) scores for clustering performance
- Number of messages processed by each model
- Results saved as `data/groups/thisiscere/metrics_conversations_thisiscere.csv`

Example output:
```csv
model,label_file,ari,n_messages
143535,labels_20250131_143535_gpt4o_thisiscere.csv,0.583,49
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.865,67
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.568,49
```

Which translates to this more readable table:

| Model | ARI Score (-1 to 1) | Messages Processed | Notes |
|-------|-------------------|-------------------|-------|
| GPT-4 | 0.583 | 49 | Moderate conversation grouping accuracy |
| DeepSeek V3 | 0.865 | 67 | Strong conversation grouping, processed more messages |
| Claude 3.5 | 0.568 | 49 | Moderate conversation grouping accuracy |

This table shows that DeepSeek V3 achieves notably better conversation grouping accuracy (ARI score) while also processing more messages. GPT-4 and Claude 3.5 show similar performance levels, both processing the same number of messages.
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

By following these steps, you'll be able to comprehensively evaluate any LLM's performance across all three core tasks: spam detection, conversation clustering, and topic labeling. The modular nature of the evaluation framework allows you to run assessments individually, making it easy to focus on specific aspects of the analysis. All results are saved in standardized CSV formats, making it simple to compare different models or track improvements over time. Remember to check the output files in your community's directory after each evaluation run for detailed metrics and analysis.

# Step 3: Show the results

After running the evaluations, the results will be stored in the community folder you're evaluating. For example, if you're evaluating the Cere Network community (which we've been using as an example throughout this guide), the results will be in `data/groups/thisiscere/`. For any other community, replace `thisiscere` with your community's name in the paths below.

<details>
<summary><strong>📚 Knowledge Base - Theory behind evaluation metrics</strong></summary>

Now that you've seen the data, let's walk through the evaluation process. The module implements a hierarchical evaluation framework with one main task and two additional tasks:

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
   ```
   User1: "How's Arbitrum's TPS compared to other L2s?"
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

Now that we've seen the model outputs that had been pre-processed for you in advance, let's examine how these predictions translate into quantitative performance metrics across our three key evaluation criteria: spam detection, conversation clustering, and topic labeling. Each metric provides unique insights into model capabilities and limitations.

## 📊 Your Metrics for Success

### Main Task

<details>
<summary><strong>Conversation Clustering Results</strong></summary>

Results will be stored as `metrics_conversations_[community_name].csv` in your community's folder.

For example, for the Cere Network community: `data/groups/thisiscere/metrics_conversations_thisiscere.csv`

This file contains:
- ARI (Adjusted Rand Index): Measure of clustering accuracy (-1 to 1)
- n_messages: Number of messages processed by each model

Example interpretation from Cere Network results:
```csv
model,label_file,ari,n_messages
143535,labels_20250131_143535_gpt4o_thisiscere.csv,0.583,49
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.865,67
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.568,49
```

Which translates to this more readable table:

| Model | ARI Score (-1 to 1) | Messages Processed | Notes |
|-------|-------------------|-------------------|-------|
| GPT-4 | 0.583 | 49 | Moderate conversation grouping accuracy |
| DeepSeek V3 | 0.865 | 67 | Strong conversation grouping, processed more messages |
| Claude 3.5 | 0.568 | 49 | Moderate conversation grouping accuracy |

This table shows that DeepSeek V3 achieves notably better conversation grouping accuracy (ARI score) while also processing more messages. GPT-4 and Claude 3.5 show similar performance levels, both processing the same number of messages.
</details>

### Additional Tasks

<details>
<summary><strong>Spam Detection Results (Optional ⭐)</strong></summary>

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

# Step 4: 🎯 Your First Challenge - Hot swap different models

## Open Source Examples

For those interested in using open-source, self-hosted models, we provide a detailed guide and example implementations in the `open_source_examples` directory. The guide covers:

<details>
<summary><strong>1. Ollama Setup & Configuration</strong></summary>

- Complete setup instructions for Ollama
- Server configuration and management
- Model installation and deployment
- Common troubleshooting steps
</details>

<details>
<summary><strong>2. Example Implementation: Mistral 7B</strong></summary>

- Full implementation of conversation detection
- Follows the same evaluation framework
- Produces compatible output formats
- Demonstrates best practices for adding new models
</details>

Check out `open_source_examples/README.md` for:
- Detailed setup instructions
- Implementation guidelines
- Output format specifications
- Troubleshooting tips
- Instructions for adding your own models

This allows you to:
1. Run evaluations with open-source models
2. Compare results with commercial models
3. Add support for new models
4. Contribute your own implementations

# Step 5: Start understanding the modifying the prompts, pre-grouping techniques, models

This is the advanced step where you'll start to understand how to modify the prompts, pre-grouping techniques, and models to improve the performance of the model.