# **SDRMp – A Text Classification & Question-Answering Pipeline**

## **Overview**
This repository, **SDRMp**, implements a **text classification and question-answering (QA) pipeline** for analyzing textual data. It combines **deep learning-based classification (DeBERTaV3)** with **question generation (LLaMA-3-8B)** to improve interpretability and explainability in natural language understanding (NLU) tasks.  

The project includes:
- **Text Classification** using transformer-based models.
- **N-gram Analysis** for extracting significant textual patterns.
- **Question Generation** based on extracted features.
- **Automated Answering Pipeline** using DeBERTa-v3-SQuAD2.

This framework is useful for **explainable AI (XAI), stress classification, domain-specific QA systems, and NLP research**.  

---

## **Repository Structure**
### **1. `classifier_modeltraining.py`**
- Implements **text classification** using the **DeBERTa-V3 model**.
- Loads and tokenizes textual datasets.
- Trains models for classification.
- Supports multi-label classification and evaluation using **F1-score, precision, and recall**.

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
- **Computational Linguistics datasets**

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
This repository supports research in:  
✅ **Explainable AI (XAI)** through **n-gram-based probing**.  
✅ **Question-Driven Model Interpretability** using **LLMs**.  
✅ **Improved Text Classification** with **automated evaluations**.  
✅ **Benchmarking QA models** across multiple datasets.  

For more details, refer to our **research paper**.

---

## **Citation**
If you use this repository, please cite:
```
@article{SDRMp2024,
  author    = {Your Name and Collaborators},
  title     = {Explainable AI for Text Classification using N-Gram Analysis and Question Generation},
  journal   = {Research Journal},
  year      = {2024}
}
```
 
