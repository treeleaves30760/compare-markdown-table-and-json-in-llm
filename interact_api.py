import os
import requests
from jinja2 import Template


class LLMInference_api:
    def __init__(self, api_url):
        """
        Initialize the model from local or Hugging Face repository.
        Args:
        - model_name (str): Path to the local model or Hugging Face model identifier.
        - use_cuda (bool): Whether to use CUDA if available.
        """
        self.api_url = api_url

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

    def _send_message_to_llm(self, message):
        """
        Send a message to the LLM and get the response.
        Args:
        - message (str): Message to send to the model.
        """
        data = {
            "prompt": message
        }
        response = requests.post(self.api_url, json=data)
        return response.json()

    def send_message(self, rag_data, question):
        """
        Send a message to the LLM and get the response.
        It will fill in the template with the data and question.
        And call the _send_message_to_llm function.

        Args:
        - rag_data (str): Data to send fill in template.
        - question (str): Question to ask the model.
        """

        return self._send_message_to_llm(self.load_chat_template('chat_template.txt', rag_data=rag_data, question=question))
