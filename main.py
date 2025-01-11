import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFProcessor:
    def __init__(self):
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            return ""
    
    def process_directory(self, directory_path, label=None):
        papers = []
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logging.error(f"Directory not found: {directory_path}")
            return papers
        
        pdf_files = list(directory_path.rglob("*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in tqdm(pdf_files, desc=f"Processing {directory_path.name}"):
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                # Create consistent paper dictionary structure
                paper_dict = {
                    'paper_id': pdf_file.stem,
                    'content': text,
                    'file_path': str(pdf_file),
                    'label': label if label is not None else 0
                }
                papers.append(paper_dict)
            else:
                logging.warning(f"No text extracted from {pdf_file}")
        
        return papers

class PaperDataset(Dataset):
    def __init__(self, papers, tokenizer, max_length=512, is_evaluation=False):
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_evaluation = is_evaluation
    
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        paper = self.papers[idx]
        
        # Ensure all required keys exist
        if not all(key in paper for key in ['content', 'paper_id']):
            raise KeyError(f"Missing required keys in paper dictionary. Required: content, paper_id. Found: {paper.keys()}")
        
        encoding = self.tokenizer(
            paper['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'paper_id': paper['paper_id'],
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if not self.is_evaluation:
            if 'label' not in paper:
                raise KeyError(f"Label missing in training data for paper_id: {paper['paper_id']}")
            item['label'] = torch.tensor(paper['label'], dtype=torch.float)
            
        return item

class PaperEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = PaperClassifier().to(device)
    
    def train(self, train_data, val_data, epochs=3, batch_size=16, lr=1e-5):
        """Train the model using the provided training and validation data."""
        # Create datasets
        train_dataset = PaperDataset(train_data, self.tokenizer)
        val_dataset = PaperDataset(val_data, self.tokenizer, is_evaluation=True)
        
        # Create data loaders
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
            
            # Validation
            self.evaluate_on_validation(val_loader)

    @torch.no_grad()
    def evaluate_on_validation(self, val_loader):
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        total_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        correct_predictions = 0
        total_samples = 0
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples if total_samples else 0
        logging.info(f"Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

class PaperDataLoader:  # Renamed from DataLoader to avoid conflict
    """Handle data loading and preprocessing"""
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.pdf_processor = PDFProcessor()
        
    def load_reference_data(self):
        """Load reference papers from both publishable and non-publishable folders"""
        logging.info("Loading reference papers...")
        
        publishable_path = self.base_path / "Reference" / "Publishable"
        non_publishable_path = self.base_path / "Reference" / "Non-Publishable"
        
        if not publishable_path.exists():
            logging.error(f"Publishable directory not found: {publishable_path}")
        if not non_publishable_path.exists():
            logging.error(f"Non-publishable directory not found: {non_publishable_path}")
        
        publishable_papers = self.pdf_processor.process_directory(
            publishable_path, 
            label=1
        )
        logging.info(f"Loaded {len(publishable_papers)} publishable papers")
        
        non_publishable_papers = self.pdf_processor.process_directory(
            non_publishable_path, 
            label=0
        )
        logging.info(f"Loaded {len(non_publishable_papers)} non-publishable papers")
        
        all_papers = publishable_papers + non_publishable_papers
        if not all_papers:
            raise ValueError("No reference papers found! Please check the directory structure and file formats.")
            
        return all_papers
    
    def load_papers_to_evaluate(self):
        """Load papers that need to be evaluated"""
        papers_path = self.base_path / "Papers"
        if not papers_path.exists():
            logging.error(f"Papers directory not found: {papers_path}")
            return []
            
        papers = self.pdf_processor.process_directory(papers_path)
        logging.info(f"Loaded {len(papers)} papers for evaluation")
        return papers


class PaperClassifier(nn.Module):
    def __init__(self, pretrained_model="allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def main():
    try:
        logging.info(f"Current working directory: {os.getcwd()}")
        
        base_path = Path("db")
        logging.info(f"Base path: {base_path.absolute()}")
        data_loader = PaperDataLoader(base_path)
        
        # Load and validate reference papers
        reference_papers = data_loader.load_reference_data()
        
        if not reference_papers:
            logging.error("No reference papers found. Please check the data directory structure.")
            return
        
        # Split and train
        train_data, val_data = train_test_split(reference_papers, test_size=0.2, random_state=42)
        
        logging.info("Initializing model...")
        evaluator = PaperEvaluator()
        
        logging.info("Training model...")
        evaluator.train(train_data, val_data)
        
        # Load papers to evaluate
        logging.info("Loading papers to evaluate...")
        papers_to_evaluate = data_loader.load_papers_to_evaluate()
        
        if not papers_to_evaluate:
            logging.error("No papers found for evaluation!")
            return
        
        # Evaluate papers with error handling
        results = []
        logging.info("Evaluating papers...")
        for paper in tqdm(papers_to_evaluate):
            try:
                result = evaluator.evaluate_paper(paper)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing paper {paper.get('paper_id', 'unknown')}: {e}")
                results.append({
                    'paper_id': paper.get('paper_id', 'unknown'),
                    'publishable': 0,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv("results.csv", index=False)
        logging.info("Results saved to results.csv")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
