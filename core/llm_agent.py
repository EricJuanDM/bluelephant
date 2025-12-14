import os
from google import genai
from core.tools.viacep_tool import consultar_cep
from core.tools.pokeapi_tool import consultar_pokemon_data

class LLMAgent:
    def __init__(self, vector_store_manager):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = 'gemini-2.5-flash' 

        self.vector_store_manager = vector_store_manager
        self.current_prompt = self._get_initial_prompt()
        self.prompt_history = [{"version": 1, "prompt": self.current_prompt, "source": "initial_config", "feedback_count": 0}]
        self.feedback_log = []

        self.tools = {
            "consultar_cep": consultar_cep,
            "consultar_pokemon_data": consultar_pokemon_data
        }
    
    def _get_initial_prompt(self):
        return (
            "Você é um assistente de IA prestativo e cordial. Sua principal função é responder "
            "perguntas do usuário e usar as ferramentas disponíveis quando necessário. "
            "Se for perguntado sobre um CEP, use a função 'consultar_cep'. "
            "Se for perguntado sobre dados de Pokémon (nome, tipo, habilidades), use 'consultar_pokemon_data'. "
            "Sempre forneça respostas claras e diretas. Se não souber a resposta, peça desculpas e sugira o uso de uma ferramenta."
        )
    
def process_query(self, query: str):
        
        contexto_rag = self.vector_store_manager.retrieve_context(query)
        
        
        system_instruction_rag = (
            f"{self.current_prompt}\n\n"
            f"--- CONTEXTO RAG PARA INFORMAÇÕES INTERNAS ---\n"
            f"{contexto_rag}"
        )
        
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[query],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction_rag, 
                tools=self.tools
            )
        )

        return response.text

def update_prompt_from_feedback(self, query: str, response: str, rating: str, suggestion: str) -> bool:
        
        new_feedback = {
            "query": query,
            "response": response,
            "rating": rating,
            "suggestion": suggestion,
            "timestamp": genai.time.time() 
        }
        self.feedback_log.append(new_feedback)
        
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

        try:
            refinement_response = self.client.models.generate_content(
                model='gemini-2.5-pro', 
                contents=[prompt_refinement_instruction]
            )
            
            new_prompt_text = refinement_response.text.strip()
            
            if new_prompt_text and new_prompt_text != self.current_prompt:
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