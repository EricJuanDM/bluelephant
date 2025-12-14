import streamlit as st
import time 

# --- IMPORTS DAS CLASSES REAIS DO CORE (Essenciais para o funcionamento) ---
from core.llm_agent import LLMAgent 
from core.vector_store import VectorStoreManager 

# ---------------------- CONFIGURAÇÃO E INICIALIZAÇÃO ----------------------

st.set_page_config(page_title="Agente de IA com Feedback Inteligente", layout="wide")
st.title("🤖 Chatbot Inteligente com Melhoria de Prompt em Tempo Real")
st.markdown("---")

@st.cache_resource
def initialize_system():
    # 1. Inicializa o Vector Store Manager (ChromaDB)
    try:
        vs_manager = VectorStoreManager()
        vs_manager.initialize_static_data() 
    except Exception as e:
        st.error(f"Erro ao inicializar Vector Store (ChromaDB). Verifique o volume Docker: {e}")
        return None
    
    # 2. Inicializa o Agente e passa o Vector Store Manager
    try:
        agent = LLMAgent(vector_store_manager=vs_manager)
    except Exception as e:
        st.error(f"Erro ao inicializar LLM Agent. Chave Gemini API configurada? Erro: {e}")
        return None
        
    return agent

agent = initialize_system()

if agent is None:
    st.warning("O sistema de IA não pôde ser iniciado. Por favor, corrija os erros acima e reinicie.")
    st.stop()


# ---------------------- GESTÃO DO ESTADO DA SESSÃO ----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = None

# ---------------------- DEFINIÇÃO DAS ABAS ----------------------

tab_chat, tab_feedback = st.tabs(["💬 Chat do Agente", "📝 Feedback e Melhoria"])

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

        with st.spinner("🤖 O agente está processando a resposta e buscando informações..."):
            try:
                # O ERRO OCORRIA AQUI! Se o LLMAgent não tivesse o método process_query.
                agent_response = agent.process_query(prompt)

                st.session_state.last_interaction = {
                    "query": prompt,
                    "response": agent_response
                }

            except Exception as e:
                # Captura qualquer falha no processamento do LLM ou Tools
                agent_response = f"Erro no Agente: Falha ao processar a pergunta. Detalhes: {e}"
                st.error(agent_response)
                
        with st.chat_message("assistant"):
            st.markdown(agent_response)
            st.session_state.messages.append({"role": "assistant", "content": agent_response})


# ==============================================================================
# --- ÁREA 2: FEEDBACK E MELHORIA ---
# ==============================================================================
with tab_feedback:
    st.header("Avalie e Sugira Melhorias para o Agente")
    
    st.subheader("1. Fornecer Feedback sobre a Última Resposta")

    interaction = st.session_state.last_interaction
    
    if interaction:
        st.info(f"Última Pergunta: **{interaction['query']}**")
        st.info(f"Última Resposta: **{interaction['response'][:150]}...**")
    else:
        st.warning("Interaja no chat primeiro para gerar feedback.")
        

    feedback_quality = st.radio(
        "Qualidade da Resposta:",
        ('Excelente (5/5)', 'Boa (4/5)', 'Razoável (3/5)', 'Ruim (2/5)', 'Incorreta (1/5)'),
        index=None,
        horizontal=True
    )
    
    feedback_suggestion = st.text_area(
        "Sugestões de Melhoria (Obrigatório para notas baixas):",
        placeholder="A resposta estava incompleta, o agente deveria ter usado a ViaCEP para a resposta.",
        height=100
    )

    def handle_feedback():
        if interaction and feedback_quality and feedback_suggestion:
            with st.spinner("🧠 Processando feedback e gerando novo prompt..."):
                try:
                    success, message = agent.update_prompt_from_feedback(
                        query=interaction['query'],
                        response=interaction['response'],
                        rating=feedback_quality,
                        suggestion=feedback_suggestion
                    )
                except Exception as e:
                    st.error(f"❌ Falha no LLM Otimizador: {e}")
                    return
            
            if success:
                st.success(f"✅ Sucesso! {message}")
                st.session_state.last_interaction = None 
            else:
                st.info(f"ℹ️ Tentativa concluída. {message}")
        else:
            st.error("Por favor, selecione a qualidade e insira uma sugestão de melhoria.")

    st.button(
        "Enviar Feedback e Atualizar Prompt", 
        on_click=handle_feedback, 
        type="primary", 
        disabled=(interaction is None)
    )

    st.markdown("---")

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