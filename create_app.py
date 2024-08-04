""" Simple local app that creates stories using LLMs models. This is a prototype that will be customized to tailor scientific stories for kids to be engaged with science. 
    Author: Victor Torres
    LLM Model used: llama-2-7b-chat.ggmlv3.q8_0.bin
    Frameworks used: Langchain, Streamlit
"""


## Import relevant libraries

import streamlit as st
from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline
# from langchain_community.llms import CTransformers

# Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("BashitAli/llama-2-7b-chat.ggmlv3.q5_K_M")



## Define response function from local LLM:

def getLLMResponse(input_text, no_words, category):
    """ This function initializes the language model,
      defines a prompt template, and generates a response based on the user’s input. 
        """
    
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    # llm = CTransformers(model = 'BashitAli/llama-2-7b-chat.ggmlv3.q5_K_M', model_type = 'llama',
    #                     config = {'max_new_tokens': 256,
    #                               'temperature': 0.01})
    
    # Create a template for the prompt

    template = """ Write a {category} on {input_text} in less than {no_words} words. Add a title to the text. """

    prompt = PromptTemplate(input_variables= ["input_text", "no_words", "category"], template = template)

        ## Now create the response from the LLama 2 Model
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who  is an expert in creating text and story telling",
        },
        {"role": "user", "content": prompt.format(category = category, input_text = input_text, no_words = no_words)},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"].split('<|assistant|>')[-1]
    return response


st.set_page_config(page_title= "Generating a story",
                   layout = 'centered',
                   initial_sidebar_state= "collapsed")

st.header("Creative Writer")

input_text = st.text_input("Provide me a topic and I will create you a wonderful text")

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('Number of words')
with col2:
    category = st.selectbox("Category",
                              ('Essays', 'Poem', 'Joke', 'Blog'),
                              index=0)
    

submit = st.button("Generate")

if submit:

    st.write(getLLMResponse(input_text, no_words, category))
