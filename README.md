# Medical-Chatbot-using-BERT-and-GPT-model

This project introduces a novel approach by combining two powerful language models, BERT and GPT, to solve the task of generating health-related responses. BERT excels at tasks like classification, feature extraction, and embedding generation, while GPT is highly effective for sequence-to-sequence (seq2seq) tasks, such as generating text. The idea here is to leverage both models for optimal performance in a medical chatbot system.

The project is divided into four main parts:
  1) Building a Feature Extractor with BERT: BERT is used to create a feature extraction model, capturing the contextual embeddings of questions and answers.
  2) Generating and Storing Embeddings: The trained feature extractor model is used to generate embeddings for all questions and answers in the dataset, which are saved as PyTorch tensors.
  3) Similarity Search with FAISS: FAISS, a library from Facebook, is used to perform similarity searches, generating a set of question-answer pairs based on similarity scores for each entry in the dataset.
  4) Fine-Tuning GPT with Question-Answer Pairs: The final step involves feeding the entire dataset, along with the paired question-answer data, into the GPT model to train it for response generation.

**The project is still in the development phase, with further fine-tuning and additional training of the models yet to be completed.**
