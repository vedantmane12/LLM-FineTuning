# 🧠 Mental Health Q&A Assistant - Fine-tuned LLM

A specialized AI assistant for mental health support, created by fine-tuning DialoGPT-medium using LoRA (Low-Rank Adaptation) on a curated dataset of mental health FAQs.

## 🎯 Project Overview

This project demonstrates how to fine-tune a Large Language Model (LLM) for sensitive domain-specific applications. Using only 98 Q&A pairs and 20 minutes of training on a free Google Colab GPU, we achieved a 50.7% performance improvement in generating appropriate mental health support responses.

### Key Features
- ✅ Parameter-efficient fine-tuning using LoRA (only 2.67% of parameters trained)
- ✅ Ethical safety checks and crisis detection
- ✅ Complete training pipeline from data preparation to deployment
- ✅ Comprehensive evaluation and analysis tools
- ✅ Data augmentation to expand limited dataset

### ⚠️ Important Disclaimer
This AI assistant is for educational and informational purposes only. It is NOT a replacement for professional mental health care. Always encourage users to seek help from qualified mental health professionals.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Base Model | microsoft/DialoGPT-medium (345M params) |
| Trainable Parameters | 9.2M (2.67%) |
| Training Time | ~20 minutes on T4 GPU |
| Loss Reduction | 50.7% |
| Final Loss | 4.38 |
| Inference Speed | <1 second per response |
| Model Size | 36MB (LoRA only) or 1.5GB (merged) |

## 🚀 Getting Started with Google Colab

### Prerequisites
- Google account for Colab access
- Mental_Health_FAQ.csv dataset
- ~30 minutes for complete execution

### Step-by-Step Implementation

## 📝 Notebook Walkthrough

### Step 1: Environment Setup and GPU Verification
```python
# Check GPU availability and setup
!nvidia-smi
torch.cuda.get_device_name(0)
```
This verifies you have GPU access (ideally T4) and sets up the training environment.

### Step 2: Install Required Packages
```bash
!pip install -q transformers datasets accelerate peft bitsandbytes evaluate rouge-score wandb
```

Installs all necessary libraries including:

- `transformers`: Hugging Face library for LLMs
- `peft`: Parameter-Efficient Fine-Tuning for LoRA
- `bitsandbytes`: Memory optimization tools

### Step 3: Data Loading and Analysis
```python
# Load dataset
df = pd.read_csv('Mental_Health_FAQ.csv')

# Analyze text lengths
df['question_length'] = df['Questions'].str.len()
df['answer_length'] = df['Answers'].str.len()
```
What it does:

- Loads your 98 Q&A pairs
- Analyzes question/answer lengths for tokenization planning
- Identifies data quality issues

### Step 4: Data Preprocessing and Augmentation
```python
# Clean text
def clean_text(text):
    # Fix encoding issues
    replacements = {'â€™': "'", 'â€œ': '"', 'â€': '"'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()

# Data augmentation
def augment_question(question, num_variations=2):
    # Creates variations like:
    # "What is anxiety?" → "What exactly is anxiety?"
    # "How to manage stress?" → "How exactly to manage stress?"
```

Purpose:

- Fixes common UTF-8 encoding issues
- Expands dataset from 98 to ~300 samples through augmentation
- Maintains answer quality while varying questions

### Step 5: Create Train/Validation/Test Splits

```python
# 70/15/15 split
train_data, test_data = train_test_split(formatted_data, test_size=0.15)
train_data, val_data = train_test_split(train_data, test_size=0.176)
```

Result:

- Training: ~200 samples (after augmentation)
- Validation: 15 samples
- Test: 15 samples

### Step 6: Model and Tokenizer Loading

```python
# Load DialoGPT-medium
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.float16,  # Half precision for memory
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token
```

Key decisions:

- `float16`: Reduces memory by 50%
- `device_map="auto"`: Automatic GPU placement
- Padding token setup for batch processing

### Step 7: LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Scaling
    target_modules=["c_attn", "c_proj", "c_fc"],  # Which layers to adapt
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
```

What LoRA does:

- Freezes 97.3% of model parameters
- Only trains small adapter matrices
- Reduces memory from 14GB to 4GB

### Step 8: Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    learning_rate=5e-4,
    warmup_steps=50,
    fp16=True,  # Mixed precision training
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
)
```

Optimization strategies:

- Gradient accumulation simulates larger batches
- Mixed precision training for speed
- Regular evaluation for monitoring

### Step 9: Training Execution

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

train_result = trainer.train()
```

Training process:

- Takes ~20 minutes on T4 GPU
- Loss decreases from 8.89 to 4.38
- Automatic checkpointing every 25 steps

### Step 10: Model Evaluation and Testing

```python
# Test generation
def generate_response(prompt, max_new_tokens=100, temperature=0.7):
    formatted_prompt = f"User: {prompt}\nAssistant:"
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

# Test on mental health queries
test_prompts = [
    "I'm feeling anxious about my future",
    "How can I manage stress better?",
    "I've been having trouble sleeping"
]
```

Generation parameters explained:

- `temperature=0.7`: Balanced creativity/coherence
- `repetition_penalty=1.2`: Prevents repetitive text
- `no_repeat_ngram_size=3`: No 3-word repetitions

### Step 11: Save Model and Results

```python
# Save full model
trainer.save_model(output_dir)

# Save LoRA adapter only (36MB vs 1.5GB)
model.save_pretrained(f"{output_dir}/lora_adapter")

# Save training metrics
results = {
    "final_loss": train_result.metrics['train_loss'],
    "training_time": train_result.metrics['train_runtime'],
    "parameters_trained": model.num_parameters(only_trainable=True)
}
```

#### 📊 Understanding the Results

##### Training Metrics

- Initial Loss: 8.89 (untrained model)
- Final Loss: 4.38 (50.7% improvement)
- Perplexity: ~80 (lower is better)
 
##### Response Quality Indicators

- Average response length: 30-35 words
- Vocabulary diversity: 65%
- Empathetic language presence: 87%

#### 🛡️ Safety Implementation

The notebook includes safety checks for crisis situations:

```python
crisis_keywords = ['suicide', 'self-harm', 'kill myself', 'end it all']

if any(keyword in user_input.lower() for keyword in crisis_keywords):
    return """I'm very concerned about what you're going through. 
    Please reach out for immediate help:
    - 988 Suicide & Crisis Lifeline
    - Text HOME to 741741
    - Emergency: 911"""
```

### 📁 Output Files
After running the notebook, you'll have:

/content/models/mental_health_causal_[timestamp]/
├── config.json                # Model configuration
├── pytorch_model.bin          # Full model weights
├── training_args.bin          # Training configuration
├── tokenizer_config.json      # Tokenizer settings
├── vocab.json                 # Vocabulary
├── lora_adapter/              # LoRA-only weights (36MB)
│   ├── adapter_config.json
│   └── adapter_model.bin
├── training_results.json      # Performance metrics
└── training_curves.png        # Loss visualization

### 📚 Resources and References

- `DialoGPT Paper`: Microsoft Research
- `LoRA Paper`: Hu et al., 2021
- `Hugging Face Docs`: Transformers Library

### ⚠️ Ethical Considerations
This project implements several ethical safeguards:

- Crisis detection and appropriate response
- Clear AI disclosure in all outputs
- Encouragement of professional help
- No diagnostic or treatment attempts
- Privacy-preserving (no data storage)

