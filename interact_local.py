import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template


class LLMInference:
    def __init__(self, model_name, use_cuda=True):
        """
        Initialize the model from local or Hugging Face repository.
        Args:
        - model_name (str): Path to the local model or Hugging Face model identifier.
        - use_cuda (bool): Whether to use CUDA if available.
        """
        self.device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=self.get_hf_token())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=self.get_hf_token()).to(self.device)

    def get_hf_token(self):
        """
        Retrieve the Hugging Face API token from an environment variable.
        """
        HF_TOKEN = os.getenv('HF_TOKEN')
        if HF_TOKEN:
            print(f"HF_TOKEN environment variable found.")
            return os.getenv('HF_TOKEN')
        else:
            print(f"No HF_TOKEN environment variable found.")
            return None

    def load_chat_template(self, filename, **kwargs):
        """
        Load a chat template and replace variables in the template.
        Args:
        - filename (str): The filename of the template in the 'prompt_template' directory.
        - kwargs (dict): Variables to replace in the template.
        """
        template_path = os.path.join('prompt_template', filename)
        with open(template_path, 'r', encoding='utf-8') as file:
            template_content = file.read()
            template = Template(template_content)
            return template.render(**kwargs)

    def _send_message_to_LLM(self, message):
        """
        Send a message to the LLM and get the response.
        Args:
        - message (str): Message to send to the model.
        """
        inputs = self.tokenizer.encode(
            message, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=256)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def send_message(self, rag_data, question, template_name='llama3_basic.prompt'):
        """
        Combine rag_data and message into template and send to the model.

        Args:
        - rag_data (str): Data for the RAG model.
        - question (str): Question to ask the model.
        - template_name (str): Name of the template file in the 'prompt_template' directory.
        """
        message = self.load_chat_template(
            filename=template_name, rag_data=rag_data, question=question)
        return self._send_message_to_LLM(message)


# Example usage:
if __name__ == "__main__":
    # Example model name, replace with your actual model path or Hugging Face model identifier
    model_name = "gpt2"
    chatbot = LLMInference(model_name)

    # Load a template and replace a placeholder
    prompt = chatbot.load_chat_template(
        'example.prompt', variable_name="OpenAI")
    print("Loaded and rendered prompt:", prompt)

    # Send a message to the model
    response = chatbot.send_message(prompt)
    print("Model response:", response)
