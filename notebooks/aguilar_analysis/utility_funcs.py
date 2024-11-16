import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sentence_transformers import SentenceTransformer

#Sentiment Classification model
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def classify_text_batch(series: pd.Series, model=model, tokenizer=tokenizer, embedding_model=embedding_model, batch_size=32):
    """
    Classifies text in batches using a pre-trained model and tokenizer, saving embeddings and labels.

    Args:
        series: A pandas Series containing the text to classify.
        model: The pre-trained model.
        tokenizer: The tokenizer used for tokenization.
        embedding_model: The embedding model.
        batch_size: The batch size for inference.

    Returns:
        A pandas DataFrame with columns for text, embeddings, and predicted class labels.
    """


    results = []
    embeddings_list = []
    for i in range(0, len(series), batch_size):
        batch = series[i:i+batch_size]
        inputs = tokenizer(batch.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            embeddings = embedding_model.encode(batch.tolist())  # Adjust max_length if needed

        predicted_class_ids = logits.argmax(dim=-1).tolist()
        results.extend([model.config.id2label[id] for id in predicted_class_ids])
        embeddings_list.extend(embeddings.tolist())

    df_results = pd.DataFrame({'text': series.tolist(), 'embeddings': embeddings_list, 'label': results})
    return df_results

def chunk_dataframe(df, chunk_size=1000):
  """
  Chunks a DataFrame into smaller DataFrames of the specified size.

  Args:
    df: The input DataFrame.
    chunk_size: The desired size of each chunk.

  Returns:
    A list of DataFrames, each with a maximum of `chunk_size` rows.
  """

  chunks = []
  for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunks.append(chunk)

  return chunks