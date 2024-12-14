# Importing necessary libraries from Hugging Face
from transformers import pipeline

# Initialize the summarization pipeline with a pre-trained model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text
input_text = """There are times when the night sky glows with bands of color. 
The bands may begin as cloud shapes and then spread into a great arc across the entire sky. 
They may fall in folds like a curtain drawn across the heavens."""

# Get the summary from the model
summary = summarizer(input_text, max_length=50, min_length=30, do_sample=False)

# Output the refined summary
refined_summary = summary[0]['summary_text']

# Display the refined summary
print("Refined Summary:", refined_summary)
