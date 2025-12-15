import os
import time
import json
from google import genai
from pydantic import BaseModel, Field # Importar Pydantic para schemas
# Importações das ferramentas (que são apenas funções Python)
from core.tools.viacep_tool import consultar_cep
from core.tools.pokeapi_tool import consultar_pokemon_data


#  SCHEMA DE SAÍDA OBRIGATÓRIO PARA O LLM OTIMIZADOR (Correção de Robustez - Ponto 4)
# Força o Gemini a retornar um JSON com este formato, garantindo que o novo prompt não seja "poluído".
class PromptSchema(BaseModel):
    """Esquema de saída forçado para o LLM Otimizador."""
    new_system_prompt: str = Field(description="O novo e melhorado Prompt do Sistema, sem introdução ou explicação. Deve ser apenas o texto puro do novo prompt.")


class LLMAgent:
    def __init__(self, vector_store_manager):
        # 1. Configuração do LLM (Gemini API)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não configurada. Verifique o .env e docker-compose.yml.")
            
        self.client = genai.Client(api_key=api_key)
        
        #  Corrigindo Hardcoding (Ponto 2): Lendo modelos do ambiente
        self.model = os.environ.get("MODEL_AGENT_CORE", 'gemini-2.5-flash') 
        self.model_optimizer = os.environ.get("MODEL_AGENT_OPTIMIZER", 'gemini-2.5-pro')


        # 2. FERRAMENTAS (TOOLS) - Inicializado antes do prompt para evitar bug de inicialização
        self._tool_map = { 
            "consultar_cep": consultar_cep,
            "consultar_pokemon_data": consultar_pokemon_data
        }
        # Lista de funções para o Gemini (API espera uma lista de objetos de função)
        self.tools_for_gemini = list(self._tool_map.values())


        # 3. Gerenciamento do Prompt e Feedback
        self.vector_store_manager = vector_store_manager
        self.current_prompt = self._get_initial_prompt()
        self.prompt_history = [{"version": 1, "prompt": self.current_prompt, "source": "initial_config", "feedback_count": 0}]
        self.feedback_log = []
    
    def _get_initial_prompt(self):
        """Define a personalidade inicial, informando ao LLM quais ferramentas ele possui."""
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
        """
        Gera a resposta do agente utilizando o prompt atual, tools e contexto (RAG).
        Orquestração Simplificada de Ferramentas (Correção de Arquitetura - Ponto 3).
        """
        
        # 1. Recuperar contexto da Vector Store (RAG)
        contexto_rag = self.vector_store_manager.retrieve_context(query)
        
        # 2. Instrução do Sistema Aprimorada
        system_instruction_rag = (
            f"{self.current_prompt}\n\n"
            f"--- CONTEXTO RAG PARA INFORMAÇÕES INTERNAS ---\n"
            f"{contexto_rag}"
        )
        
        # Iniciar a lista de conversação com a query do usuário
        contents = [query]
        
        # Loop para orquestrar tool calls (máximo 3 iterações)
        for i in range(3): 
            # 3. Chamada ao LLM (passando o histórico e resultados de tool)
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents, 
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction_rag,
                    tools=self.tools_for_gemini 
                )
            )

            # 4. Se o LLM não chamou uma ferramenta, ele gerou a resposta final
            if not response.function_calls:
                return response.text 

            # 5. Executa todas as chamadas de ferramenta solicitadas
            tool_response_parts = []
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args)
                
                # Executa a função Python real
                print(f"-> Agente chamando Tool: {tool_name} com args: {tool_args}")
                result = self._tool_map[tool_name](**tool_args) 
                
                # Adiciona o resultado da execução da ferramenta para o próximo turno do LLM
                tool_response_parts.append(genai.types.Part.from_function_response(
                    name=tool_name,
                    response=result
                ))
            
            # Adiciona a resposta da ferramenta à lista de conteúdo para o próximo turno
            contents.extend(tool_response_parts)

        # Retorno de segurança se atingir o limite de iterações
        return "O agente atingiu o limite de chamadas de ferramentas e não conseguiu gerar uma resposta."

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
        
        # 2. Constrói a instrução para o LLM Otimizador
        prompt_refinement_instruction = (
            # ... (Instruções detalhadas para o LLM Otimizador) ...
        )

        # 3. Usa o LLM (Otimizador) para gerar o novo prompt, forçando o JSON
        try:
            refinement_response = self.client.models.generate_content(
                model=self.model_optimizer, 
                contents=[prompt_refinement_instruction],
                #  FORÇANDO A SAÍDA ESTRUTURADA (Correção Ponto 4)
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PromptSchema,
                )
            )
            
            # Parseia o JSON garantido e pega o valor do prompt
            response_json = json.loads(refinement_response.text)
            new_prompt_text = response_json['new_system_prompt'].strip()
            
            if new_prompt_text and new_prompt_text != self.current_prompt:
                # 4. Atualiza o prompt e o histórico (lógica de versão)
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
            return False, f"Erro ao gerar novo prompt com LLM Otimizador: {e}"