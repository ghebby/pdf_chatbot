import streamlit as st
from streamlit_chat import message as st_message
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain import VectorDBQA
# from langchain.llms import OpenAI
# from langchain.vectorstores import Chroma
from langchain.llms import OpenAIChat
from PIL import Image
import pickle
import os

pdf_output = None

st.set_page_config(page_title="PDFBot", page_icon="🤖", layout="wide")

aenish_pic = Image.open('./aenish_pic.jpeg')

#sidebar
feedback_url = "https://forms.gle/LwqkLVuxdggkjFU59"
share_url = "https://t.co/vQmbVhNyoH"

with st.sidebar:
    st.markdown("# About 🙏")
    st.markdown(
        "Introducing our revolutionary 🤖 PDF chatbot! Say goodbye to endless scrolling \n"
        "and searching through long PDF documents. Our chatbot allows you to have a\n"
        "conversation with your PDF and get the information you need in seconds.🤝 \n"
        "With easy integration into your existing systems, our PDF chatbot is the perfect solution for businesses and individuals alike. \n"
        "Try it out today and experience the future of PDF interaction.📚\n"
        )
    st.markdown(
        "Unlike chatGPT, PDFBOT can't make stuff up\n"
        "and will only answer from injected knowlege 📖 \n"
    )
    st.markdown("---")
    st.markdown("🧑‍💻 A side project by Aenish Shrestha")
    st.image(aenish_pic, width=60)
    st.markdown("---")
    st.markdown("😊 Give feedback [here](%s)" %feedback_url)
    st.markdown("---")
    st.markdown("🔎 Find Me [here](%s)" %share_url)
    st.markdown("---")




st.title("Turn Your PDF To Chatbot")

st.write("This app will help you answer questions based on your PDF")

# st.subheader("Step 1: 🔑 Setup your OpenAI API Key")
# ask for a user text input
# user_openai_api_key = st.text_input("Enter your OpenAI API Key",placeholder="OPENAI_API_KEY",value="")
# st.write("You can get yours from here - https://beta.openai.com/account/api-keys")

st.subheader("Step 1: 📤 Upload your PDF file")
uploaded_file = st.file_uploader("Choose a file")

user_openai_api_key = st.secrets["user_api_key"]

if user_openai_api_key is None:
    st.warning("Error : Api Key Not Found ❌")

if uploaded_file is None:
    st.warning("Error : Pdf File Not Found ❌")
# if uploaded_file is not None and user_openai_api_key is not None:
if uploaded_file is not None and user_openai_api_key is not None:

    reader = PdfReader(uploaded_file)
     # Get the file name
    filename = uploaded_file.name
    # Write the bytes of the file to a file on the local machine
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Show a success message
#     st.success(f"Saved file '{filename}' to disk. Please Wait , File Is Processing.")
    raw_text = ''

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

#     st.write("your file is ready to be processed and now is splitting into text, please wait...")

    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size = 750,
        chunk_overlap  = 0
    )
    texts = text_splitter.split_text(raw_text)

#     st.write("File is split into chunks: ", len(texts))
    pdf_output = st.success("🤖 Bot Is Online ✅")

    embeddings = OpenAIEmbeddings(openai_api_key=user_openai_api_key)

# if filename is not None:
    # st.write("Your {} Bot Is Online 🟢".format(filename))


def generate_output(user_prompt):
    # docsearch = Chroma.from_documents(texts,embeddings)
    # qa = VectorDBQA.from_chain_type(llm=OpenAI(),chain_type="stuff",vectorstore=docsearch)

    with open("foo.pkl", 'wb') as f:
        pickle.dump(embeddings, f)

    with open("foo.pkl", 'rb') as f:
        new_docsearch = pickle.load(f)

    docsearch = FAISS.from_texts(texts, new_docsearch)
    docs = docsearch.similarity_search(user_prompt)
    response = docs[0].page_content
    
    #OPENAI CHATGPT MODEL
    prefix_messages = [
    {"role": "system", "content": "You are a helpful AI bibliophile Tutor.I want you to act as a document that I am having a conversation with. Your name is AI Assistant. You will provide me with answers from the given text. If the answer is not included in the text, say exactly Hmm, I am not sure. NEVER mention the provided text, remember you are the provided text I am having a chat with. Never break character."},
    {"role": "user", "content": "I am student, I want to learn from this document/file/book , so you can give me answers based on this file/document/book, also if i aks you to do mathematical conversions,translations you can do it but answer it in professional way. I want you to help me with this file/document/book with your 100% effort."},
    {"role": "assistant", "content": "Thats awesome, what do you want to know about"},]


    chain = load_qa_chain(OpenAIChat(openai_api_key=user_openai_api_key,temperature=0,prefix_messages=prefix_messages), chain_type="stuff")
    ai_output = chain.run(input_documents=docs, question=user_prompt)
    
    #OPENAI DAVINICI MODEL
#     chain = load_qa_chain(OpenAI(openai_api_key=user_openai_api_key,temperature=0), chain_type="stuff")
#     Base_Prompt = '''You are an AI assistant that provides answers from the given document and only from the document! If the answer is not in the document, say "Hmm, I am not sure". Never try to come up with an answer if the info is not in the document. Reply in the same language as the question.''' 
#     Final_Question = Base_Prompt+" "+user_prompt
    
#     ai_output = chain.run(input_documents=docs, question=Final_Question)
    # # query="Minimum Dimension OF Kitchen"
    # ai_output = qa.run(user_prompt)
    
    ai_output = ai_output.replace("\n\n--","").replace("\n--","").strip()
    return ai_output,response
    

#enable Chat 
if "history" not in st.session_state:
    st.session_state.history = []




def generate_answer():
    
    user_message = st.session_state.input_text
    tokenizer,result= generate_output(user_message)
    # inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
    # result = model.generate(**inputs)
    # message_bot = tokenizer.decode(
    #     result[0], skip_special_tokens=True
    # )  # .replace("<s>", "").replace("</s>", "")
    message_bot = tokenizer


    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})

if pdf_output:
    st.text_input("Type A Specific Message","Who are you?", key="input_text")
    if st.button("Tell me about it", type="primary"):
        generate_answer()
else :
    st.write("📚 Please Upload Your PDF.")
    

for chat in st.session_state.history[::-1]:
#     st_message(**chat)  # unpacking
# Fix Duplicate streamlit keys
    st_message(key='input_text', **chat) 
 
    
    
    
    
    
    
    
    


