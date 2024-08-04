"""
This is the main file for the compare markdown table and other table-like data structure project. 
This file will be used to run the LLM and interact with it.
"""

# Set up the LLM
from interact import LLMInference

LLM = LLMInference("meta-llama/Meta-Llama-3-8B-Instruct")
