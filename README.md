# **GUIDEQ: Framework for Guided Questioning for progressive informational collection and classification**

## **Overview**
Our work, **GUIDEQ**, presents a novel framework for asking guided questions to further progress a partial information. We leverage the explainability derived from the classifier model for along with LLMs for asking
guided questions to further enhance the information. This further information helps in more accurate classification of a text. GUIDEQ derives the most significant key-words representative of a label using occlusions. We develop GUIDEQ’s prompting strategy for guided questions based on the top-3 classifier label outputs and the significant words, to seek specific and relevant information, and classify in a targeted manner. 

The project includes:
- **Text Classification** using transformer-based models.
- **N-gram Analysis** for extracting significant textual patterns.
- **Question Generation** based on extracted features.
- **Automated Answering Pipeline** using DeBERTa-v3-SQuAD2.
 

---

## **Repository Structure**
### **1. `classifier_modeltraining.py`**
- Implements **text classification** using the **DeBERTa-V3 model**.
- Loads and tokenizes textual datasets.
- Trains models for classification.
- Supports multi-class classification and evaluation using **F1-score, precision, and recall**.

### **2. `que_gen.py` (Question Generation)**
- Generates **context-aware questions** using **LLaMA-3-8B-Instruct**.
- Uses **top n-gram features** extracted from classified text to guide question generation.
- Helps in **interpretable AI** by probing the model’s reasoning process.

### **3. `guided_words.py` (N-Gram Analysis)**
- Extracts significant **n-gram phrases** (uni-, bi-, tri-grams) from classified texts.
- Performs **occlusion-based importance analysis** to find influential words.
- Outputs **top-ranked n-grams** to guide question formation.

### **4. `answers_qapipeline.py` (Answer Generation & Evaluation)**
- Uses **DeBERTa-v3-large-SQuAD2** for **automated question answering**.
- Takes context and generated questions to produce answers.
- Scores answer confidence and **compares different question strategies (baseline vs. LLM-generated questions).**
- Outputs **final evaluation CSVs**.

---

## **Datasets**
The repository processes datasets from **multiple sources**, supporting:
- **20NG (20 Newsgroups)**
- **S2D (Sentence-to-Document)**
- **CNews (Chinese News)**
- **Stress-related text classification**
- **SALAD (Scientific Article Labeling And Discovery)**
- **DBPedia**
 

Each dataset undergoes **preprocessing, classification, question generation, and evaluation**.

---

## **How to Use**
### **1. Setup Environment**
```bash
pip install -r requirements.txt
```
Ensure that required transformers and deep-learning libraries are installed.

### **2. Run Text Classification**
```bash
python classifier_modeltraining.py
```
Trains a **DeBERTa-V3 classifier** for dataset-based text categorization.

### **3. Generate Questions**
```bash
python que_gen.py
```
Generates **context-aware questions** to improve model interpretability.

### **4. Perform N-Gram Analysis**
```bash
python guided_words.py
```
Extracts **important textual patterns** that influence classification.

### **5. Evaluate QA Pipeline**
```bash
python answers_qapipeline.py
```
Runs the **QA pipeline**, storing **answers and confidence scores**.

---

## **Research Contributions**
- **Introduction of GUIDEQ**: A novel framework designed to generate guided questions for improving classification of partial information.

- **Leverage Explainable Keywords**: Utilizes explainable keywords derived from classifier models to enhance question generation.

- **LLM-based Question Generation**: Combines large language models (LLMs) for generating questions, facilitating better context understanding.

- **Superior Performance**: Demonstrates superior performance across multiple datasets, with improvements in F1 scores of up to 22% compared to baseline methods.

- **High-Quality Question Generation**: GUIDEQ generates context-relevant, high-quality questions that outperformed baseline methods in various tests.

- **Multiturn Interaction Capability**: Effective in multiturn interactions, allowing for more dynamic and refined information retrieval.

- **Flexibility with N-gram Approaches**: Adaptable to different n-gram approaches for keyword generation, enhancing the framework’s flexibility for diverse applications.

- **Potential for Real-World Applications**: Shows promise for real-world applications in information retrieval and classification tasks, particularly in scenarios involving partial or incomplete information.


For more details, refer to our **research paper**.



---

## **Citation**
If you use this repository, please cite:
```
@article{SDRMp2024,
  author    = {Priya Mishra, Suraj Racha, Kaustubh Ponkshe,Adit Akarsh and Ganesh Ramakrishnan},
  title     = {GUIDEQ: Framework for Guided Questioning for progressive informational collection and classification},
  journal   = {NACCL},
  year      = {2024}
}
```
 
