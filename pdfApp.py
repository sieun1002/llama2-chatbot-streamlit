import json
import datetime
import requests
import streamlit as st
import tempfile
import os
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

# ChatPDF 클래스 정의
class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama2:latest")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise. [/INST] </s>

            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path):
        docs = PyPDFLoader(file_path=pdf_file_path).load() #PDF 파일 로딩
        chunks = self.text_splitter.split_documents(docs) #텍스트 분할
        chunks = filter_complex_metadata(chunks) # 메타데이터 필터링
        self.vector_store = Chroma.from_documents( #벡터 스토어 생성
            documents=chunks, embedding=FastEmbedEmbeddings()
        )
        self.retriever = self.vector_store.as_retriever( #검색기 설정 
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3, #최대 반환 분서 수
                "score_threshold": 0.5, #유사성 점수 임계값
            },
        )
        self.chain = ( #처리 체인 설정
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | self.prompt 
            | self.model 
            | StrOutputParser()
        )

    def ask(self, query):
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

# 챗봇 모델에 대한 요청 함수
def run_prompt(model, prompt):
    r = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': prompt
        }
    )
    return [json.loads(row) for row in r.text.splitlines()]

# Streamlit 애플리케이션 구성
st.title("Llama2 chatbot")

# PDF 파일 업로드
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
if uploaded_file is not None:
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    chat_pdf = ChatPDF()
    chat_pdf.ingest(temp_file_path)
    st.success("PDF file uploaded and processed.")

    # 임시 파일 삭제 (선택적)
    os.remove(temp_file_path)

# LLaMA 2 모델 지정
model = "llama2:latest"

# 채팅 이력 초기화 및 표시
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 챗봇 응답 생성 및 표시
def generate_chat_response(prompt):
    if uploaded_file is not None:
        # PDF 파일이 업로드된 경우
        return chat_pdf.ask(prompt)
    else:
        # PDF 파일이 업로드되지 않은 경우
        response = run_prompt(model, prompt)
        return "\n".join([line.get('response', '') for line in response if line.get('done') != True])

# 사용자 입력 처리
if prompt := st.chat_input('What is up?'):
    user_message = f'''
        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}:

        {prompt}
        '''
    st.chat_message('user').text(user_message)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # 챗봇 응답 생성 및 표시
    with st.spinner('Querying LLaMA 2...'):
        chat_response = generate_chat_response(prompt)
        
    if chat_response:
        chat_response = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}:\n\n{chat_response}"
        with st.chat_message('assistant'):
            st.markdown(chat_response)
            st.session_state.messages.append({'role': 'chatbot', 'content': chat_response})