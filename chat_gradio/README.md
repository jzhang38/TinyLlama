## Tinyllama Chatbot Implementation with Gradio

We offer an easy way to interact with Tinyllama. This guide explains how to set up a local Gradio demo for a chatbot using TinyLlama.
(A demo is also available on the Hugging Face Space [TinyLlama/tinyllama_chatbot](https://huggingface.co/spaces/TinyLlama/tinyllama-chat)) or Colab [colab](https://colab.research.google.com/drive/1qAuL5wTIa-USaNBu8DH35KQtICTnuLsy?usp=sharing).

### Requirements
* Python>=3.8
* PyTorch>=2.0
* Transformers>=4.34.0
* Gradio>=4.13.0

### Installation
`pip install -r requirements.txt`

### Usage

`python TinyLlama/chat_gradio/app.py`

* After running it, open the local URL displayed in your terminal in your web browser. (For server setup, use SSH local port forwarding with the command: `ssh -L [local port]:localhost:[remote port] [username]@[server address]`.)
* Interact with the chatbot by typing questions or commands.


**Note:** The chatbot's performance may vary based on your system's hardware. Ensure your system meets the above requirements for optimal experience.
