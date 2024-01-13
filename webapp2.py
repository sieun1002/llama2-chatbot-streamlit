import streamlit as st 
import requests 

#Streamlit 앱의 타이틀 설정
st.title("llama2 chatbot 서비스")

#세션 상태 초기화
if "messages" not in st.session_state:
  st.session_state.messages = []

#이전 메시지들을 화면에 표시 
for message in st.session_state.messages: 
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

#채팅 입력
if prompt := st.chat_input("Your message"):
  #사용자 메시지 저장 및 표시
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)
  
  #Ollama 서버에 요청 보내기
  try:
    response = requests.post("http://localhost:11434/api/generate", json={"model": "llama2",
  "prompt":prompt})
    if response.status_code == 200:
      full_response = response.json().get("reply") 
      st.session_state.messages.append({"role": "assistant", "content": full_response})
      with st.chat_message("assistant"):
        st.markdown(full_response)
    
    else:
      st.error("Error: Unable to get response from the server")
  
  except Exception as e:
    st.error(f"An error occurred: {e}")