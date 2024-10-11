import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set device to GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained or fine-tuned GPT-2 model and tokenizer
model_name = "gpt2"  # Replace with the path of your fine-tuned model if needed
model = torch.load("Trained_models/chatbot_model.pth", map_location=torch.device('cpu'))
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Move the model to the appropriate device
model.to(device)
model.eval()  # Set the model to evaluation mode for inference

def generate_response(input_text, max_length=100):
    """
    Generate a response using the GPT-2 model based on the user input.

    :param input_text: User input string
    :param max_length: Maximum number of tokens to generate
    :return: Generated response string
    """
    # Encode the input text into tokens
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate response from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length,
                                pad_token_id=0,
                                do_sample=True, top_p=0.9, temperature=0.7)

    # Decode the generated tokens into a string
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text[len(input_text):]  # Remove the input from the response

# Interactive chatbot loop
def chatbot():
    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit the chatbot
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Generate and print the chatbot response
        response = generate_response(user_input.lower())
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()