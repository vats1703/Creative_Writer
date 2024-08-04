# Creative Writers and LLMs

Creative Writers and LLMs is a small local application that leverages open-source language models to generate text based on a given topic, type, and word limit. This application is built using Python and makes use of the LangChain, Streamlit frameworks, and the Transformers library.

## Features

- **Flexible LLM Usage:** While this app is compatible with various language models, it currently considers using [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
- **Four Writing Categories:** Generate text in the form of poems, jokes, essays, or blogs.
- **Interactive UI:** Utilize an intuitive interface powered by Streamlit.

## Screenshots

![Screenshot of Output Example in the Interface](path/to/your/screenshot1.png)

![Screenshot of Output Example](path/to/your/screenshot2.png)

## Requirements

Ensure you have the following Python packages installed (as listed in `requirements.txt`):

- `sentence-transformers`
- `uvicorn`
- `ctransformers`
- `langchain`
- `python-box`
- `streamlit`

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/creative-writers-llms.git
   cd creative-writers-llms
