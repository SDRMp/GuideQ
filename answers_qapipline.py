import os
import pandas as pd
from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/deberta-v3-large-squad2",device= 2)

def process_csv(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df = df.fillna('- - -')
    print(file_path)
    # Ensure the necessary columns are present
    required_columns = [ 'second_half', 'questionnk', 'questionllm']
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {file_path}: Missing required columns.")
        return

    # # Create empty lists to store results
    answers, scores = [], []
    answersnk, scoresnk = [], []
    answersllm, scoresllm = [], []

    # Apply the QA pipeline to each row
    for index, row in df.iterrows():
        question = row['question']
        questionnk = row['questionnk']
        questionllm = row['questionllm']
        context = row['second_half']
 
 
        if question is not None:
            result = qa_pipeline(question=question, context=context)
            answers.append(result['answer'])
            scores.append(result['score'])
        else:
            answers.append(None)
            scores.append(None)

        # Second question
        if questionnk is not None:
            resultnk = qa_pipeline(question=questionnk, context=context)
            answersnk.append(resultnk['answer'])
            scoresnk.append(resultnk['score'])
        else:
            answersnk.append(None)
            scoresnk.append(None)

        # Third question
        if questionllm is not None:
            resultllm = qa_pipeline(question=questionllm, context=context)
            answersllm.append(resultllm['answer'])
            scoresllm.append(resultllm['score'])
        else:
            answersllm.append(None)
            scoresllm.append(None)

 

    # Add the results to the DataFrame
    df['answer'] = answers
    df['score'] = scores
    df['answernk'] = answersnk
    df['scorenk'] = scoresnk
    df['answerllm'] = answersllm
    df['scorellm'] = scoresllm

    # Save the modified DataFrame with a new filename
    new_file_path = os.path.join(os.path.dirname(file_path), 'finaleval_' + os.path.basename(file_path))
    df.to_csv(new_file_path, index=False)
    print(f"Processed and saved {new_file_path}")

dstypes = ['20NG', 's2d', 'cnews','stress','salad','dbp','comp']
mnames = ['bert','debertaV3']



def traverse_and_process(folder_path): 
    for i in range(3):
        for dstype in dstypes:
            for mname in mnames:
                ngram_length = i+1
                print(f"./{dstype}/{ngram_length}grams_{mname}/{dstype}_{ngram_length}Q.csv")
                p = (f"./{dstype}/{ngram_length}grams_{mname}/{dstype}_{ngram_length}Q.csv")
                process_csv(p)

# Define the folder path
folder_path = '.'

# Traverse the folder and process each CSV file
traverse_and_process(folder_path)