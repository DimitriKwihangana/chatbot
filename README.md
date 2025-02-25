# 🌾🤖 Agriculture Chatbot using GPT-2 Transformer

## 🚀 Project Overview
This project implements a chatbot using a fine-tuned **GPT-2 Transformer** model from Hugging Face. The chatbot is designed to assist stakeholders in agriculture by providing relevant information, answering questions, and building capacity through AI-driven conversations. By leveraging AI, this chatbot enhances knowledge sharing and decision-making among farmers, agribusiness professionals, and policymakers.

## 📂 Dataset
- **Format**: JSON
- **Structure**: A collection of question-answer pairs related to agriculture
- **Preprocessing**: Tokenization, normalization, and handling missing values
- **Purpose**: Improve chatbot's ability to provide accurate and insightful responses in the agricultural domain

## 🏗️ Model & Training
- **Pre-trained Model**: [GPT-2](https://huggingface.co/gpt2)
- **Fine-tuning**: Custom dataset with TensorFlow
- **Hyperparameter Tuning**: Learning rate, batch size, optimizer selection, training epochs
- **Loss Function**: Cross-entropy loss
- **Evaluation Metrics**: BLEU Score, F1-Score, Perplexity

## 📊 Performance Evaluation
- **BLEU Score**: Measures text fluency and coherence
- **F1-Score**: Evaluates response accuracy
- **Perplexity**: Assesses how well the model predicts the next word
- **Qualitative Testing**: Interactive testing to ensure meaningful responses specific to agriculture



## 📜 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gpt2-agriculture-chatbot.git
   cd gpt2-agriculture-chatbot
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the pre-trained model and fine-tune it on your dataset**:
   ```python
   python train.py
   ```
4. **Run the chatbot interface**:
   ```python
   python app.py
   ```

## 📌 Features
- ✅ Fine-tuned GPT-2 model for agricultural support
- ✅ JSON-based question-answer dataset focused on agriculture
- ✅ TensorFlow implementation with Hugging Face `transformers`
- ✅ Web-based, CLI, and API-based interaction options
- ✅ Performance evaluation using NLP metrics

## 💡 Future Enhancements
- 🔹 Improve response quality with larger datasets
- 🔹 Implement memory for contextual understanding
- 🔹 Optimize model for lower latency responses
- 🔹 Expand dataset with expert-verified agricultural knowledge
