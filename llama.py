import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm

model_name = "meta-llama/Llama-2-13b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
llama = LlamaForCausalLM.from_pretrained(model_name, do_sample=True).cuda()