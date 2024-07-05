#THIS FILE WORKS, generates the 3 graphs

# Commented out IPython magic to ensure Python compatibility.
#%pip install datasets tiktoken transformers[torch] accelerate emoji==0.6.0 -U

# Step 1: Create requirements.txt
requirements = """
accelerate==0.31.0
aiohttp==3.9.5
aiosignal==1.3.1
altair==5.3.0
async-timeout==4.0.3
attrs==23.2.0
certifi==2024.6.2
charset-normalizer==3.3.2
contourpy==1.2.1
cycler==0.12.1
datasets==2.20.0
dill==0.3.8
emoji==0.6.0
filelock==3.13.1
fonttools==4.53.0
frozenlist==1.4.1
fsspec==2024.2.0
gensim==4.3.2
huggingface-hub==0.23.4
idna==3.7
Jinja2==3.1.3
joblib==1.4.2
jsonschema==4.22.0
jsonschema-specifications==2023.12.1
kiwisolver==1.4.5
MarkupSafe==2.1.5
matplotlib==3.9.0
mpmath==1.3.0
multidict==6.0.5
multiprocess==0.70.16
networkx==3.2.1
numpy==1.26.4
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==8.7.0.84
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.20.5
nvidia-nvtx-cu11==11.8.86
p2j==1.3.2
packaging==24.1
pandas==2.2.2
pillow==10.3.0
pip==22.0.2
plotly==5.22.0
psutil==6.0.0
pyarrow==16.1.0
pyarrow-hotfix==0.6
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
referencing==0.35.1
regex==2024.5.15
requests==2.32.3
rpds-py==0.18.1
safetensors==0.4.3
scikit-learn==1.5.0
scipy==1.10.1
seaborn==0.13.2
setuptools==59.6.0
six==1.16.0
smart-open==7.0.4
sympy==1.12
tenacity==8.4.1
threadpoolctl==3.5.0
tiktoken==0.7.0
tokenizers==0.19.1
toolz==0.12.1
tqdm==4.66.4
transformers==4.41.2
triton==2.3.1
typing_extensions==4.9.0
tzdata==2024.1
urllib3==2.2.2
wrapt==1.16.0
xxhash==3.4.1
yarl==1.9.4
"""

""" This is for using notebooks (i created a venv with the above packages) """
# with open('requirements.txt', 'w') as f:
#     f.write(requirements)

# Step 2: Install the packages using pip
# %pip install -r requirements.txt
# %pip install torch --index-url https://download.pytorch.org/whl/cu118

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import tiktoken
import logging

""" For notebooks """
# from google.colab import drive, files
# drive.mount('/content/drive')

# from huggingface_hub import notebook_login
# notebook_login() # this will ask for hugging face token (very important)

"""Enable logging"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Function to create GPT-4 tokenizer using tiktoken"""
def create_gpt4_tokenizer():
    enc = tiktoken.get_encoding("cl100k_base")
    return enc

"""Initialize tokenizer"""
tokenizer = create_gpt4_tokenizer()

"""Function to read text file and label data"""
def read_text_file(filepath, cyber_label):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [{'text': line.strip(), 'cyber_label': cyber_label} for line in lines]

"""Load bias lexicons"""
def load_lexicon(filepaths):
    lexicon = set()
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as file:
            lexicon.update([line.strip() for line in file.readlines()])
    return lexicon

bias_lexicon = load_lexicon(['drive_content/en.txt', 'drive_content/en2.txt'])

# Read the datasets
cyber_data = read_text_file('drive_content/8000age.txt', cyber_label=1)
nocber_data = read_text_file('drive_content/8000notcb.txt', cyber_label=0)

"""Combine the datasets"""
data = cyber_data + nocber_data

"""Create a DataFrame"""
df = pd.DataFrame(data)
logger.info(f"Dataset shape: {df.shape}")
logger.info(df.head())

"""Split the dataset into train and test sets"""
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
logger.info(f"Training set shape: {train_df.shape}")
logger.info(f"Test set shape: {test_df.shape}")
logger.info(train_df.head())

"""Convert the DataFrame to a Dataset"""
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

"""Simple Bias Detection Method"""
def simple_bias_detection(text):
    words = text.split()
    bias_count = sum(1 for word in words if word.lower() in bias_lexicon)
    return bias_count / len(words) if len(words) > 0 else 0

"""Apply simple bias detection method"""
df['simple_bias'] = df['text'].apply(simple_bias_detection)

"""Convert bias scores to binary labels (threshold = 0.5)"""
df['bias_label'] = (df['simple_bias'] > 0.5).astype(int)

"""Update the dataset"""
df['labels'] = df.apply(lambda x: [x['cyber_label'], x['bias_label']], axis=1)
df = df[['text', 'labels']]
logger.info(df.head())

"""Split the updated dataset into train and test sets"""
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

"""Initialize the Hugging Face AutoTokenizer"""
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

"""Function to tokenize data using AutoTokenizer"""
def tokenize_function(examples):
    return hf_tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

"""Data collator"""
data_collator = DataCollatorWithPadding(hf_tokenizer)

"""Define the custom multi-label classification model class"""
class MultiLabelClassificationModel(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(MultiLabelClassificationModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels).to(device)
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, logits
        return logits

"""Function to train and evaluate a model"""
def train_and_evaluate_model(model_name, token=None):
    global main_tokenizer, trainer, model, tokenized_datasets
    auth_token = {'token': token} if token else {}
    main_tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_token)
    # main_tokenizer.model_max_length = 130
    print(main_tokenizer.model_max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")

    model = MultiLabelClassificationModel(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        use_cpu=False,
        logging_dir='./logs',
        logging_steps=10,
        report_to='none',
        no_cuda=False
    )

    def compute_metrics(p):
        predictions, labels = p
        preds = torch.sigmoid(torch.tensor(predictions)).numpy()
        preds = (preds > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='samples')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=main_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    logger.info(f"Training model: {model_name}")
    trainer.train()

    # Evaluate the model
    logger.info(f"Evaluating model: {model_name}")
    evaluation_results = trainer.evaluate()
    logger.info(f"Evaluation results: {evaluation_results}")

    return evaluation_results

"""List of models to evaluate"""
models = [
    "vinai/bertweet-base",
    "cardiffnlp/twitter-roberta-base",
    "mistralai/Mistral-7B-v0.3",
    "distilbert-base-uncased",
    "microsoft/MiniLM-L12-H384-uncased",
    "d4data/bias-detection-model"
]

"""Dictionary to store evaluation results"""
results = {}

"""Hugging Face token"""
token = "hf_xxxxxxx" # TODO update hugging face token here

"""Train and evaluate each model"""
for model_name in models:
    logger.info(f"Evaluating model: {model_name}")
    try:
        results[model_name] = train_and_evaluate_model(model_name, token=token)
    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {e}")

"""Plot results using Plotly"""
def plot_results(results):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "F1 Score"))

    for model_name, metrics in results.items():
        if 'eval_accuracy' in metrics:
            fig.add_trace(go.Bar(x=[model_name], y=[metrics['eval_accuracy']], name=model_name), row=1, col=1)
        if 'eval_f1' in metrics:
            fig.add_trace(go.Bar(x=[model_name], y=[metrics['eval_f1']], name=model_name), row=1, col=2)

    fig.update_layout(height=600, width=1000, title_text="Model Comparison")
    fig.show()

plot_results(results)

"""Train Word2Vec model"""
sentences = [text.split() for text in df['text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, negative=5, epochs=10)

"""Function to visualize word vectors using PCA"""
def visualize_vectors(model, words, title="Word Vectors"):
    word_vectors = np.array([model.wv[word] for word in words if word in model.wv])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(word_vectors)

    fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1], text=words)
    fig.update_layout(title=title, xaxis_title="PC1", yaxis_title="PC2")
    fig.show()

"""Visualization of word vectors"""
# Visualization of word vectors
bias_words = list(bias_lexicon)[:50]

# Filter out words that are not in the model's vocabulary
bias_words = [word for word in bias_words if word in w2v_model.wv]
visualize_vectors(w2v_model, bias_words, title="PCA of Bias Word Vectors")

"""Visualization of token embeddings"""
sample_text = "This is a sample sentence for token embedding visualization."
tokens = hf_tokenizer.tokenize(sample_text)
token_embeddings = hf_tokenizer.encode(sample_text, return_tensors='pt')

# Move token_embeddings to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token_embeddings = token_embeddings.to(device)

"""Run the model to get token embeddings"""
with torch.no_grad():
    outputs = model.base_model(token_embeddings)
    embeddings = outputs.logits.squeeze().cpu().numpy()

# Ensure embeddings is a 2D array
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)
elif embeddings.ndim == 2 and embeddings.shape[0] == 1:
    embeddings = embeddings.reshape(-1, embeddings.shape[1])

# Check the shape of embeddings
n_samples, n_features = embeddings.shape
n_components = min(2, n_samples, n_features)

# Check for NaNs or infinities and handle them
if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
    print("Embeddings contain NaNs or infinities. Handling them by replacing with zeros.")
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

"""Apply PCA"""
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(embeddings)

"""Plot token embeddings using Plotly"""
# Ensure we have the correct shape for plotting
if n_components == 1:
    principal_components = np.hstack((principal_components, np.zeros((principal_components.shape[0], 1))))

# Match the length of tokens with principal_components
tokens = tokens[:principal_components.shape[0]]

# Plot the PCA result
fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1], text=tokens)
fig.update_layout(title="PCA of Token Embeddings", xaxis_title="PC1", yaxis_title="PC2")
fig.show()