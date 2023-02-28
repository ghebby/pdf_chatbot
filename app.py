import streamlit as st
from streamlit_chat import message as st_message
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import VectorDBQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import pickle
import os

st.title("PDF to chatbot")

st.write("This app will help you to create a chatbot from a PDF file")

# st.subheader("Step 1: 🔑 Setup your OpenAI API Key")
# ask for a user text input
# user_openai_api_key = st.text_input("Enter your OpenAI API Key",placeholder="OPENAI_API_KEY",value="")
# st.write("You can get yours from here - https://beta.openai.com/account/api-keys")

st.subheader("Step 2: 📤 Upload your PDF file")
uploaded_file = st.file_uploader("Choose a file")

user_openai_api_key = st.secrets["user_api_key"]

if uploaded_file is None and user_openai_api_key is None:
    st.write("Error : ❌ No Pdf File Found")
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
        chunk_size = 1000,
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

    chain = load_qa_chain(OpenAI(openai_api_key=user_openai_api_key,temperature=0), chain_type="stuff")
    ai_output = chain.run(input_documents=docs, question=user_prompt)

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
#     st.text_input("Type A Specific Message", key="input_text", on_change=generate_answer)
    input_text = st.text_input("Type A Specific Message")
    if st.button("Tell me about it", type="primary"):
        generate_answer()

for chat in st.session_state.history[::-1]:
    st_message(**chat)  # unpacking
