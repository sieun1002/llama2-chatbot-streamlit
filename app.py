import json
import datetime
import requests
import streamlit as st

st.title("Llama2 chatbot")

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

# LLaMA 2 모델 지정
model = "llama2:latest" 

# 채팅 이력 초기화 및 표시
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 사용자 입력 처리
if prompt := st.chat_input('What is up?'):
    st.chat_message('user').text(f'''
        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}:

        {prompt}
        ''')
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # 챗봇 응답 생성 및 표시
    with st.spinner(f'querying LLaMA 2...'):
        response = run_prompt(model, prompt)

    chat_response = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}:\n\n"
    with st.chat_message('assistant'):
        for line in response:
            if line['done'] != True and 'response' in line:
                chat_response += line['response']
        st.markdown(chat_response)
        st.session_state.messages.append({'role': 'chatbot', 'content': chat_response})
