import streamlit as st
import time 
import os
import requests 

# --- IMPORTS DAS CLASSES REAIS DO CORE ---
from core.llm_agent import LLMAgent 
from core.vector_store import VectorStoreManager 

# ---------------------- CONFIGURAÇÃO E INICIALIZAÇÃO ----------------------

st.set_page_config(page_title="Agente de IA com Feedback Inteligente", layout="wide")
st.title(" Chatbot Inteligente com Melhoria de Prompt em Tempo Real")
st.markdown("---")

# Usa o cache para inicializar o sistema apenas uma vez por sessão
@st.cache_resource
def initialize_system():
    # 1. Inicializa o Vector Store Manager (ChromaDB)
    try:
        vs_manager = VectorStoreManager()
        vs_manager.initialize_static_data() # Carrega os dados RAG
    except Exception as e:
        st.error(f"Erro ao inicializar Vector Store: {e}")
        return None
    
    # 2. Inicializa o Agente LLM
    try:
        agent = LLMAgent(vector_store_manager=vs_manager)
    except Exception as e:
        st.error(f"Erro ao inicializar LLM Agent: {e}")
        return None
        
    return agent

agent = initialize_system()

if agent is None:
    st.warning("O sistema de IA não pôde ser iniciado. Por favor, corrija os erros acima e reinicie.")
    st.stop()


# ---------------------- GESTÃO DO ESTADO DA SESSÃO ----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

#  CORREÇÃO DE UX (Ponto 1): Armazena todas as interações não avaliadas
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = [] 

# ---------------------- DEFINIÇÃO DAS ABAS ----------------------

tab_chat, tab_feedback = st.tabs([" Chat do Agente", " Feedback e Melhoria"])

# ==============================================================================
# --- ÁREA 1: CHAT DO AGENTE ---
# ==============================================================================
with tab_chat:
    st.header("Converse com o Agente")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Pergunte algo ou solicite uma ação.")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(" O agente está processando a resposta e buscando informações..."):
            try:
                agent_response = agent.process_query(prompt)
            except Exception as e:
                agent_response = f"Erro no Agente: Falha ao processar a pergunta. Detalhes: {e}"
                st.error(agent_response)
                
        with st.chat_message("assistant"):
            st.markdown(agent_response)
            st.session_state.messages.append({"role": "assistant", "content": agent_response})
            
            #  ARMAZENAMENTO DA INTERAÇÃO PARA FEEDBACK (Ponto 1)
            st.session_state.feedback_history.append({
                "id": len(st.session_state.feedback_history) + 1,
                "query": prompt,
                "response": agent_response
            })


# ==============================================================================
# --- ÁREA 2: FEEDBACK E MELHORIA (UX Aprimorada) ---
# ==============================================================================
with tab_feedback:
    st.header("Avalie e Sugira Melhorias para o Agente")
    
    st.subheader("1. Fornecer Feedback sobre a Interação")

    # Mapeia o histórico para um dicionário amigável para o selectbox
    feedback_options = {
        f"#{item['id']} - {item['query'][:50]}..." : item 
        for item in st.session_state.feedback_history
    }
    
    if not feedback_options:
        st.warning("Interaja no chat primeiro para ter respostas disponíveis para feedback.")
        selected_key = None
        selected_interaction = None
    else:
        # Seletor para escolher qual interação avaliar
        selected_key = st.selectbox(
            "Selecione a Interação para Avaliar:",
            options=list(feedback_options.keys()),
            index=0
        )
        selected_interaction = feedback_options[selected_key]
        
        # Exibe a interação selecionada
        st.info(f"Pergunta: **{selected_interaction['query']}**")
        st.info(f"Resposta do Agente: **{selected_interaction['response'][:200]}...**")
    
    
    # Restante dos campos (Qualidade e Sugestão)
    feedback_quality = st.radio(
        "Qualidade da Resposta:",
        ('Excelente (5/5)', 'Boa (4/5)', 'Razoável (3/5)', 'Ruim (2/5)', 'Incorreta (1/5)'),
        index=None,
        horizontal=True
    )
    
    feedback_suggestion = st.text_area(
        "Sugestões de Melhoria (Obrigatório):",
        placeholder="A resposta estava incompleta, o agente deveria ter usado a ViaCEP para a resposta.",
        height=100
    )

    def handle_feedback(interaction_data):
        """Função chamada ao clicar no botão de feedback."""
        if interaction_data and feedback_quality and feedback_suggestion:
            with st.spinner(" Processando feedback e gerando novo prompt..."):
                try:
                    success, message = agent.update_prompt_from_feedback(
                        query=interaction_data['query'],
                        response=interaction_data['response'],
                        rating=feedback_quality,
                        suggestion=feedback_suggestion
                    )
                except Exception as e:
                    st.error(f" Falha no LLM Otimizador: {e}")
                    return
            
            if success:
                st.success(f" Sucesso! {message}")
                
                #  Remove a interação do histórico APÓS o sucesso (Ponto 1)
                id_to_remove = interaction_data['id']
                st.session_state.feedback_history = [
                    item for item in st.session_state.feedback_history if item['id'] != id_to_remove
                ]
            else:
                st.info(f"ℹ️ Tentativa concluída. {message}")
        else:
            st.error("Por favor, selecione a qualidade e insira uma sugestão de melhoria.")

    # Botão de Feedback
    st.button(
        "Enviar Feedback e Atualizar Prompt", 
        on_click=handle_feedback, 
        args=(selected_interaction,),
        type="primary", 
        disabled=(selected_interaction is None)
    )

    st.markdown("---")

    # 3. Visualização do Prompt Atual e Histórico
    st.subheader("2. Status e Histórico de Prompts")
    
    st.text(f"Versão Atual: {len(agent.prompt_history)}")
    st.code(agent.current_prompt, language="markdown")
    
    with st.expander(f"Histórico de {len(agent.prompt_history)} Versões de Prompt"):
        for entry in reversed(agent.prompt_history):
            st.markdown(f"**Versão {entry['version']}** - *Fonte: {entry['source']}*")
            st.code(entry['prompt'], language="markdown")
            if 'based_on_feedback' in entry:
                st.caption(f"Motivação: '{entry['based_on_feedback']['suggestion']}'")
            st.divider()