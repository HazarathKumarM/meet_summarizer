from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load the pre-trained model and tokenizer from Hugging Face
model = TFAutoModelForSeq2SeqLM.from_pretrained('t5-small')
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Save the model in TensorFlow SavedModel format
model.save('./t5-small-savedmodel', save_format='tf')  # Saves model as a SavedModel directory
tokenizer.save_pretrained('./t5-small-savedmodel')  # Saves tokenizer
