import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

TRAIN_DATASET_PATH = "dataset-exemplos-completo.csv" #Format: ID;Text;Label
INFERENCE_INPUT_PATH = "subm3.csv"  # Format: ID;Text
INFERENCE_OUTPUT_PATH = "predictions.csv"   #Expected Output: ID;Label
MODEL_OUTPUT_DIR = "./production_ai_classifier"
VAL_DATASET_PATH="validation.csv"

#   Determine the best device available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE.upper()}")

def train_setfit_classifier():
    #   Load Training Data
    df_train = pd.read_csv(TRAIN_DATASET_PATH, sep=';')
    df_train = df_train.rename(columns={"Text": "text", "Label": "label"})
    train_dataset = Dataset.from_pandas(df_train)
    
    #   Load Validation Data
    df_val = pd.read_csv(VAL_DATASET_PATH, sep=';')
    df_val = df_val.rename(columns={"Text": "text", "Label": "label"})
    val_dataset = Dataset.from_pandas(df_val)
    
    #   Load model and send it to the GPU
    model = SetFitModel.from_pretrained("BAAI/bge-large-en-v1.5", max_length=512).to(DEVICE)
    
    args = TrainingArguments(
        batch_size=8,
        num_epochs=1,
        seed=42,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    metrics = trainer.evaluate()
    print(f"Final Model Metrics: {metrics}")
    
    model.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model successfully saved to {MODEL_OUTPUT_DIR}")

def run_batch_inference():
    #   Load model and send it to the GPU
    model = SetFitModel.from_pretrained(MODEL_OUTPUT_DIR).to(DEVICE)
    
    df_input = pd.read_csv(INFERENCE_INPUT_PATH, sep=';')
    texts = df_input["Text"].tolist()
    
    predictions = model.predict(texts)
    
    df_output = pd.DataFrame({
        "ID": df_input["ID"],
        "Label": predictions
    })
    
    df_output.to_csv(INFERENCE_OUTPUT_PATH, sep=';', index=False)
    print(f"Inference complete. Results successfully saved to {INFERENCE_OUTPUT_PATH}")

if __name__ == "__main__":
    train_setfit_classifier()
    run_batch_inference()