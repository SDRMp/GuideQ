import os
import re
import torch
import html
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from collections import defaultdict
from nltk.util import ngrams
import torch.nn.functional as F


# Load models and tokenizer
def load_model_and_tokenizer(model_path, num_labels=None):
    """
    Load and return a model and tokenizer for sequence classification or causal language modeling.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if num_labels:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
    return tokenizer, model


# Set up CUDA device
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


# Functions for Question Generation
def generate_text(prompt, model, tokenizer):
    """
    Generates text based on a prompt using the provided model and tokenizer.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(inputs['input_ids'], max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_question(partial_info, labels, keywords, model, tokenizer):
    """
    Generates a question to differentiate between categories using partial info and category keywords.
    """
    input_text = f"""

    You are an medical Expert. You are provided with partial information along with the top-3 categories where this information could belong. Each category also has a list of keywords that represent the characteristic content covered by the category. Your task is to ask an information-seeking question based on the partial information and the category keywords such that when answered, one of the categories can be selected with confidence.

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
 
    Partial information: {partial_info}
    """
    
    # Format the categories and keywords
    for label, keyword in zip(labels, keywords):
        input_text += f"Category: {label}, Keywords: {keyword}\n"
    
    # Generate the question using the model
    generated_text = generate_text(input_text, model, tokenizer)
    
    # Extract and clean the generated question
    all_sentences = re.findall(r'"(.*?)"', generated_text)
    remaining_sentences = all_sentences[3:]
    question = "None"
    
    # Select the first question with more than 15 words
    for sentence in remaining_sentences:
        if len(sentence.split()) > 15:
            question = sentence
            break
    
    return generated_text, question


# QA Helper
def get_answer(question, context, qa_pipeline):
    """
    Uses the QA pipeline to get an answer for the given question and context.
    """
    return qa_pipeline(question=question, context=context)


 

def aggregate_and_filter_positive_attributions(ngram_attributions, threshold=0):
    """
    Filters n-grams with importance greater than the specified threshold.
    """
    return {ngram: value for ngram, value in ngram_attributions.items() if value > threshold}


def select_top_ngrams(ngram_importances, top_n=5):
    """
    Selects the top N important n-grams based on their occlusion importance.
    """
    sorted_ngrams = sorted(ngram_importances.items(), key=lambda item: item[1], reverse=True)
    return sorted_ngrams[:top_n]


# DataFrame Processing
def parse_sets(set_string):
    """
    Parses a string representation of a set and returns the corresponding set.
    """
    decoded_string = html.unescape(set_string)
    set_strings = re.findall(r'\{.*?\}', decoded_string)
    actual_sets = []
    
    for set_str in set_strings:
        elements = set_str.strip('{}').split('\', \'')
        cleaned_elements = {elem.strip('\'') for elem in elements}
        actual_sets.append(cleaned_elements)
    
    return actual_sets


def generate_questions(df, model, tokenizer):
    """
    Generates questions for each row in the dataframe using the provided model and tokenizer.
    """
    questions = []
    for index, row in df.iterrows():
        question_text, generated_question = generate_question(row['first_half'], row['top3_predicted_labels'], row['significant_words'], model, tokenizer)
        questions.append(generated_question)
    df['generated_question'] = questions
    return df


def get_answers_and_confidence(df, qa_pipeline):
    """
    Gets answers and confidence scores for each generated question in the dataframe.
    """
    answers = []
    confidence_scores = []
    for index, row in df.iterrows():
        answer_result = get_answer(row['generated_question'], row['second_half'], qa_pipeline)
        answers.append(answer_result['answer'])
        confidence_scores.append(answer_result['score'])
    df['answer'] = answers
    df['confidence_score'] = confidence_scores
    return df


def perform_occlusion(df, tokenizer, model):
    """
    Perform occlusion for each row and update the 'significant_words1' column.
    """
    for index, row in df.iterrows():
        ngram_attributions = occlusion(row['generated_question'], row['label'], 2, tokenizer, model)
        positive_attributions = aggregate_and_filter_positive_attributions(ngram_attributions)
        top_ngrams = select_top_ngrams(positive_attributions, top_n=5)
        current_keywords = [kw for kw in row['significant_words'] if kw not in [ngram for ngram, _ in top_ngrams]]
        df.at[index, 'significant_words1'] = current_keywords
    return df


# Main execution
def main(df, model_path, num_labels, qa_pipeline):
    """
    Main function to process the dataframe with question generation, answering, and occlusion.
    """
    # Load models and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_path, num_labels)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Generate questions
    df = generate_questions(df, model, tokenizer)
    
    # Get answers and confidence scores
    df = get_answers_and_confidence(df, qa_pipeline)
    
    # Perform occlusion
    df = perform_occlusion(df, tokenizer, model)
    
    return df

 
df = main(df, model_path="your_model_path", num_labels=3, qa_pipeline=qa_pipeline)
print(df.head())


def process_row(row, turns=3):
    """
    Processes a single row through three turns of guided questioning.
    This version removes used keywords directly after each question generation step.
    """
    current_keywords = row['significant_words']
    initial_partial_info = row['first_half']  # Preserve initial partial information
    current_context = row['second_half']  # Context remains unchanged
    
    for turn in range(1, turns + 1):
        # Generate Question based on partial info, categories, and refined guiding words
        question_text, generated_question = generate_question(initial_partial_info, row['top3_predicted_labels'], current_keywords)
        print(f"Turn {turn} Question: {generated_question}")
        
        # Extract Answer from the model using the generated question and context
        answer_result = get_answer(generated_question, current_context, qa_pipeline)
        extracted_answer = answer_result['answer']
        confidence_score = answer_result['score']
        print(f"Turn {turn} Answer: {extracted_answer} (Score: {confidence_score})")
        
        # Remove used keywords from the current_keywords list by direct word matching
        used_keywords = [kw for kw in current_keywords if kw in generated_question]
        current_keywords = [kw for kw in current_keywords if kw not in used_keywords]
        
        # Combine partial info, guiding words, and extracted answer for next turn
        initial_partial_info = f"{initial_partial_info} {extracted_answer}"  # Maintain history
    
    return row


# Apply the process_row function to each row in the DataFrame
df = df.apply(process_row, axis=1)
