import os
from google import genai
from core.tools.viacep_tool import consultar_cep
from core.tools.pokeapi_tool import consultar_pokemon_data
import time

class LLMAgent:
    def __init__(self, vector_store_manager):
        # 1. Configuração do LLM (Gemini API)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # Esta exceção fará com que o st.error no app.py seja exibido
            raise ValueError("GEMINI_API_KEY não configurada. Verifique o docker-compose.yml.")
            
        self.client = genai.Client(api_key=api_key)
        self.model = os.environ.get("MODEL_AGENT_CORE", 'gemini-2.5-flash') # Usa fallback
        self.model_optimizer = os.environ.get("MODEL_AGENT_OPTIMIZER", 'gemini-2.5-pro') 
        
        # 2. FERRAMENTAS (TOOLS) - DEFINA ISSO PRIMEIRO!
        # Dicionário para execução interna (Python)
        self._tool_map = { 
            "consultar_cep": consultar_cep,
            "consultar_pokemon_data": consultar_pokemon_data
        }
        # Lista de funções para o Gemini (API)
        self.tools_for_gemini = list(self._tool_map.values())


        # 3. Gerenciamento do Prompt e Feedback (Agora seguro para chamar _get_initial_prompt)
        self.vector_store_manager = vector_store_manager
        # Chama o prompt DEPOIS que as ferramentas estão definidas
        self.current_prompt = self._get_initial_prompt() 
        self.prompt_history = [{"version": 1, "prompt": self.current_prompt, "source": "initial_config", "feedback_count": 0}]
        self.feedback_log = []
    
    def _get_initial_prompt(self):
        """Define a personalidade e as capacidades iniciais do agente, incluindo as ferramentas."""
        tool_names = list(self._tool_map.keys())
        tool_list_str = ", ".join(tool_names)

        return (
            "Você é um assistente de IA prestativo e cordial. Sua principal função é responder "
            "perguntas do usuário e usar as ferramentas disponíveis quando necessário. "
            f"Suas ferramentas disponíveis são: {tool_list_str}. "
            "Sempre forneça respostas claras e diretas. Priorize o contexto RAG para regras internas. "
            "Se não souber a resposta, peça desculpas e sugira o uso de uma ferramenta."
        )

    def process_query(self, query: str):
        """Gera a resposta do agente utilizando o prompt atual, tools e contexto (RAG)."""
        
        # 1. Recuperar contexto da Vector Store (RAG)
        contexto_rag = self.vector_store_manager.retrieve_context(query)
        
        # 2. Instrução do Sistema Aprimorada
        system_instruction_rag = (
            f"{self.current_prompt}\n\n"
            f"--- CONTEXTO RAG PARA INFORMAÇÕES INTERNAS ---\n"
            f"{contexto_rag}"
        )
        
        # 3. Primeira Chamada ao LLM
        response = self.client.models.generate_content(
            model=self.model,
            contents=[query],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction_rag,
                tools=self.tools_for_gemini # Usa a lista de funções que o Gemini espera
            )
        )

        # 4. Lógica para execução de Tool Calls (Função Call)
        if response.function_calls:
            tool_response_parts = []
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args)
                
                # Executa a função Python correspondente usando o _tool_map
                result = self._tool_map[tool_name](**tool_args) 
                
                tool_response_parts.append(genai.types.Part.from_function_response(
                    name=tool_name,
                    response=result
                ))
            
            # Envia o resultado da tool de volta para o LLM para gerar a resposta final
            response = self.client.models.generate_content(
                model=self.model,
                contents=[query] + tool_response_parts, 
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction_rag
                )
            )

        return response.text

    def update_prompt_from_feedback(self, query: str, response: str, rating: str, suggestion: str) -> tuple[bool, str]:
        """Processa o feedback e atualiza o prompt do agente dinamicamente."""
        
        # 1. Loga o feedback
        new_feedback = {
            "query": query,
            "response": response,
            "rating": rating,
            "suggestion": suggestion,
            "timestamp": time.time()
        }
        self.feedback_log.append(new_feedback)
        
        # 2. Constrói a instrução para o LLM de "Melhoria de Prompt"
        prompt_refinement_instruction = (
            "Você é um Otimizador de Prompts de IA. Sua tarefa é analisar o feedback do usuário "
            "e reescrever o 'Prompt Atual do Sistema' para garantir que o erro ou falha identificado seja corrigido em interações futuras. "
            "Mantenha a personalidade inicial do assistente. Foco apenas na melhoria do comportamento. "
            "\n\n--- DADOS DE FEEDBACK ---\n"
            f"Pergunta do Usuário: {query}\n"
            f"Resposta Anterior do Agente: {response}\n"
            f"Avaliação/Nota: {rating}\n"
            f"Sugestão do Usuário: {suggestion}\n"
            "\n--- PROMPT ATUAL DO SISTEMA ---\n"
            f"{self.current_prompt}"
            "\n\nCom base nas informações acima, forneça SOMENTE o NOVO Prompt do Sistema melhorado, sem introdução ou explicação adicional."
        )

        # 3. Usa o LLM (Gemini-2.5-pro) para gerar o novo prompt
        try:
            refinement_response = self.client.models.generate_content(
                model=self.model_optimizer, 
                contents=[prompt_refinement_instruction]
            )
            
            new_prompt_text = refinement_response.text.strip()
            
            if new_prompt_text and new_prompt_text != self.current_prompt:
                # 4. Atualiza o prompt e o histórico
                self.current_prompt = new_prompt_text
                new_version = len(self.prompt_history) + 1
                self.prompt_history.append({
                    "version": new_version,
                    "prompt": self.current_prompt,
                    "source": f"feedback_v{new_version}",
                    "based_on_feedback": new_feedback
                })
                
                return True, f"Prompt atualizado para a Versão {new_version} com sucesso!"
            
            return False, "Otimizador não sugeriu alteração significativa no prompt."

        except Exception as e:
            return False, f"Erro ao gerar novo prompt com LLM: {e}"