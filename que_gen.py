"""
Text Classification & Question Generation Pipeline
"""

import os
import re
import html
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer
)
from colorama import Fore, Style

# Configuration
class Config:
    # Experiment parameters
    NGRAM_LENGTH = 3
    DATASET_TYPE = 'stress'
    MODEL_NAME = 'debertaV3'
    LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    TOP_N = 3
    
    # Path configurations
    BASE_DIR = Path("./om")
    DATA_DIR = BASE_DIR / "om4/stress"
    OUTPUT_DIR = BASE_DIR / f"om3/{DATASET_TYPE}/{NGRAM_LENGTH}grams_{MODEL_NAME}"
    MODEL_DIR = BASE_DIR / f"om5/{DATASET_TYPE}/{MODEL_NAME}_{DATASET_TYPE}"
    
    # File names
    OUTPUT_KEYS_CSV = OUTPUT_DIR / f"{DATASET_TYPE}_{NGRAM_LENGTH}keys.csv"
    OUTPUT_QUESTIONS_CSV = OUTPUT_DIR / f"{DATASET_TYPE}_{NGRAM_LENGTH}que.csv"
    
    # Model config
    CUDA_VISIBLE_DEVICES = '1'
    HF_TOKEN = 'token'

def setup_environment(config: Config):
    """Initialize environment and directories"""
    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES
    torch.cuda.empty_cache()
    
    # Create directories
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.OUTPUT_DIR / 'processing.log'),
            logging.StreamHandler()
        ]
    )

class QuestionGenerator:
    """Handles question generation using LLAMA model"""
    
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(config.LLAMA_MODEL_NAME)
        self.model = torch.nn.DataParallel(self.model).to('cuda')
        self.config = config
        
    def parse_significant_words(self, set_string: str) -> List[str]:
        """Parse stored set strings from CSV"""
        decoded = html.unescape(set_string)
        set_strings = re.findall(r'\{.*?\}', decoded)
        return [
            {elem.strip("'") for elem in s.strip('{}').split(', ')}
            for s in set_strings
        ]
    
    def generate_question(self, partial_info: str, labels: List[str], keywords: List[str]) -> str:
        """Generate information-seeking question using LLAMA"""
        prompt = self._build_prompt(partial_info, labels, keywords)
        response = self._generate_response(prompt)
        return self._extract_question(response)
    
    def _build_prompt(self, partial_info: str, labels: List[str], keywords: List[str]) -> str:
        """Construct the prompt template"""
        prompt_template = """
    You are an AI Expert. You are provided with partial information along with the top-3 categories where this information could belong. Each category also has a list of keywords that represent the characteristic content covered by the category. Your task is to ask an information-seeking question based on the partial information and the category keywords such that when answered, one of the categories can be selected with confidence.

    Follow the following thinking strategy:
    First, eliminate the categories that are not probable based on the given information. Identify the main context of the partial information and see if similar content matches any of the keywords in a category. If it doesn't, then the category can be taken out of consideration.
    Now generate a question. This question should further probe for information that will help refine the identification of the most likely category. Your question should strategically use the keywords tied to each potential category, aiming to effectively differentiate between them.

    Strictly follow the format shown in examples for output generation. Double quote the final question.

    Here are a few examples to understand better:

    Example 1:
    Partial information: I constantly sneeze and have a dry cough.

    Category: Allergy, Keywords: {headache, coughing, wet, sneeze, pain}
    Category: Diabetes, Keywords: {severe, feet, skin, rashes, infection}
    Category: Common Cold, Keywords: {swollen, cough, body, shivery, ache, dry}

    Note:
    Sneeze and dry cough are the main subjects of the partial information. Coughing is present in Allergy and common cold, but cough or sneeze is not present in Diabetes. Therefore, Diabetes can't be a possible label. Only two labels—Allergy and Common Cold—are considered. The keywords suggest that knowing about symptoms like headache, body pain, shivery, etc., will help refine the classification into one of the labels.

    Therefore only two labels, namely:
    Category: Allergy
    Category: Common Cold
    are used to form a question.

    QUESTION: "Besides fever, are you experiencing symptoms such as cough, severe headaches, localized pain, or inflammation? Also, can you describe the pattern of your fever—is it continuous or does it occur in intervals?"

    Example 2:
    Partial information: The software keeps crashing.

    Category: Software Bug, Keywords: {crash, error, bug, glitch}
    Category: User Error, Keywords: {instructions, setup, incorrect, usage}
    Category: Hardware Issue, Keywords: {overheating, components, failure, malfunction}

    Note:
    The main subject of the partial information is the software crash. The keyword 'crash' is directly related to Software Bug but could also be indirectly related to User Error and Hardware Issue. However, to differentiate, asking about the conditions under which the crash happens or if any error messages appear could help narrow down the correct category.

    Therefore, all three categories, namely:
    Category: Software Bug
    Category: User Error
    Category: Hardware Issue
    are used to form a question.

    QUESTION: "When the software crashes, do you receive any specific error messages, or does it happen during particular tasks? Have you noticed any hardware malfunctions or overheating before the crashes?"

    Example 3:
    Partial information: The car is making a strange noise.

    Category: Engine Problem, Keywords: {noise, misfire, engine, smoke}
    Category: Tire Issue, Keywords: {flat, noise, pressure, alignment}
    Category: Transmission Issue, Keywords: {shifting, noise, gears, slipping}

    Note:
    The main subject of the partial information is the strange noise. The keyword 'noise' is present in all three categories—Engine Problem, Tire Issue, Transmission Issue. Knowing more about the type of noise and when it occurs can help identify the correct category.

    Therefore, all three categories, namely:
    Category: Engine Problem
    Category: Tire Issue
    Category: Transmission Issue
    are used to form a question.

    QUESTION: "Can you describe the noise in more detail? Is it a grinding, squealing, or clicking sound? Does it happen while driving, when shifting gears, or when the car is stationary?"


    ***
    
    Now generate note and QUESTION for:
        
    """   
        return prompt_template
    
    def _generate_response(self, prompt: str) -> str:
        """Generate text from LLAMA model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = self.model.module.generate(
            inputs.input_ids, 
            max_new_tokens=200
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _extract_question(self, response: str) -> str:
        """Extract generated question from model response"""
        matches = re.findall(r'"([^"]+)"', response)
        for sentence in matches:
            if len(sentence.split()) > 10:
                return sentence
        return "None"

def main():
    """Main execution pipeline"""
    config = Config()
    setup_environment(config)
    
    # Initialize components
    qgen = QuestionGenerator(config)
    
    # Load processed data
    logging.info(f"Loading data from {config.OUTPUT_KEYS_CSV}")
    df = pd.read_csv(config.OUTPUT_KEYS_CSV)
    df['significant_words'] = df['significant_words'].apply(qgen.parse_significant_words)
    
    # Generate questions
    logging.info("Starting question generation...")
    df['question'] = None
    
    for idx in range(len(df)):
        try:
            result = qgen.generate_question(
                partial_info=df.loc[idx, 'first_half'],
                labels=df.loc[idx, 'top3_predicted_labels'],
                keywords=df.loc[idx, 'significant_words']
            )
            df.at[idx, 'question'] = result
            logging.info(f"Processed row {idx+1}/{len(df)}")
        except Exception as e:
            logging.error(f"Error processing row {idx}: {str(e)}")
    
    # Save results
    df.to_csv(config.OUTPUT_QUESTIONS_CSV, index=False)
    logging.info(f"Saved questions to {config.OUTPUT_QUESTIONS_CSV}")

if __name__ == "__main__":
    main()