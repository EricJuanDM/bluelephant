import streamlit as st
import time 
class VectorStoreManager:
    def __init__(self):
        print("VectorStoreManager Mock inicializado.")
    def retrieve_context(self, query):
        return "Contexto mock de regras internas: A política de RH é estrita e não permite perguntas pessoais."
    def initialize_static_data(self):
        pass

class LLMAgent:
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.current_prompt = self._get_initial_prompt()
        self.prompt_history = [{"version": 1, "prompt": self.current_prompt, "source": "initial_config", "feedback_count": 0}]
        self.feedback_log = []

    def _get_initial_prompt(self):
        return "Você é um assistente de IA prestativo e cordial. Use as ferramentas ViaCEP e PokéAPI."

    def process_query(self, query: str):
        time.sleep(1.5) 
        if "cep" in query.lower():
            return "Resposta simulada: Usei a ViaCEP. O CEP 01001-000 pertence à Praça da Sé, São Paulo."
        elif "pokemon" in query.lower() or "pikachu" in query.lower():
            return "Resposta simulada: Usei a PokéAPI. Pikachu é do tipo Elétrico e sua habilidade principal é Static."
        else:
            context = self.vector_store_manager.retrieve_context(query)
            return f"Resposta simulada: Não usei tools. Contexto RAG: {context}. Respondi à sua pergunta sobre '{query}'."

    def update_prompt_from_feedback(self, query: str, response: str, rating: str, suggestion: str):
        time.sleep(2)
        new_version = len(self.prompt_history) + 1
        new_prompt = f"Versão {new_version} do Prompt: O agente foi corrigido para dar mais detalhes sobre {suggestion.split()[0]} e manter a cordialidade."
        
        self.current_prompt = new_prompt
        self.prompt_history.append({
            "version": new_version, 
            "prompt": new_prompt, 
            "source": f"feedback_v{new_version}", 
            "based_on_feedback": {"rating": rating, "suggestion": suggestion}
        })
        return True, f"Prompt atualizado para a Versão {new_version} com base no feedback. Novo prompt: {new_prompt}"



st.set_page_config(page_title="Agente de IA com Feedback Inteligente", layout="wide")
st.title("🤖 Chatbot Inteligente com Melhoria de Prompt em Tempo Real")
st.markdown("---")

@st.cache_resource
def initialize_system():
    vs_manager = VectorStoreManager()
    vs_manager.initialize_static_data() 
    
    agent = LLMAgent(vector_store_manager=vs_manager)
    return agent

agent = initialize_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = None


tab_chat, tab_feedback = st.tabs(["💬 Chat do Agente", "📝 Feedback e Melhoria"])


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
            agent_response = agent.process_query(prompt)

            st.session_state.last_interaction = {
                "query": prompt,
                "response": agent_response
            }

        with st.chat_message("assistant"):
            st.markdown(agent_response)
            st.session_state.messages.append({"role": "assistant", "content": agent_response})


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
        "Sugestões de Melhoria:",
        placeholder="A resposta estava incompleta, o agente deveria ter usado a ViaCEP para a resposta.",
        height=100
    )

    def handle_feedback():
        if interaction and feedback_quality and feedback_suggestion:
            with st.spinner("🧠 Processando feedback e gerando novo prompt..."):
                success, message = agent.update_prompt_from_feedback(
                    query=interaction['query'],
                    response=interaction['response'],
                    rating=feedback_quality,
                    suggestion=feedback_suggestion
                )
            
            if success:
                st.success(f"✅ Sucesso! {message}")
                st.session_state.last_interaction = None
            else:
                st.error(f" Falha na atualização do prompt: {message}")
        else:
            st.error("Por favor, selecione a qualidade e insira uma sugestão de melhoria.")

    st.button("Enviar Feedback e Atualizar Prompt", on_click=handle_feedback, type="primary", disabled=(interaction is None))

    st.markdown("---")

    st.subheader("2. Status e Histórico de Prompts")
    
    st.text(f"Versão Atual: {len(agent.prompt_history)}")
    st.code(agent.current_prompt, language="markdown")
    
    with st.expander(f"Histórico de {len(agent.prompt_history)} Versões de Prompt"):
        for entry in reversed(agent.prompt_history):
            st.markdown(f"**Versão {entry['version']}** - *Fonte: {entry['source']}*")
            st.code(entry['prompt'], language="markdown")
            if 'based_on_feedback' in entry:
                st.caption(f"Baseado no feedback: '{entry['based_on_feedback']['suggestion']}'")
            st.divider()