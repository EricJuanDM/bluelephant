import chromadb
import os

class VectorStoreManager:
    """Gerencia a conexão e operações de RAG usando ChromaDB."""
    
    def __init__(self, collection_name="chatbot_context_data"):
        # 1. Configuração do cliente ChromaDB
        # Usa um diretório persistente para garantir que os dados não se percam 
        # após a reinicialização, o que é importante para o Docker.
        self.chroma_dir = "./chroma_data"
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # Inicializa o cliente e a coleção
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        
        # Escolha do Embedding Model (necessário para o ChromaDB)
        # O ChromaDB geralmente utiliza um modelo de embedding padrão ou pode ser configurado 
        # com um modelo da HuggingFace ou OpenAI/Gemini. Usaremos o padrão do Chroma para simplificar a dockerização.
        try:
            # Garante que a coleção existe ou a cria
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"Erro ao inicializar ChromaDB: {e}")
            raise
            
        print(f"Vector Store '{collection_name}' inicializada com {self.collection.count()} documentos.")

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """Adiciona documentos à Vector Store."""
        # Se você tiver conteúdo estático (ex: regras de negócio) a ser adicionado
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Adicionados {len(documents)} documentos à coleção.")
        except Exception as e:
            print(f"Erro ao adicionar documentos: {e}")

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """Busca contexto relevante para uma query do usuário."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Formata os resultados recuperados
            context = " ".join(results['documents'][0]) if results and results.get('documents') else "Nenhum contexto relevante encontrado na base."
            return context
        except Exception as e:
            print(f"Erro ao recuperar contexto: {e}")
            return "Erro ao acessar a Vector Store."

    def initialize_static_data(self):
        """Preenche a Vector Store com um pequeno conjunto de dados estáticos (Simulação)."""
        if self.collection.count() == 0:
            print("Preenchendo a Vector Store com dados iniciais...")
            self.add_documents(
                documents=[
                    "A política de devolução da empresa é de 30 dias após a compra, somente com o recibo original.",
                    "O horário de atendimento ao cliente é de segunda a sexta, das 9h às 18h (Horário de Brasília).",
                    "Em caso de dúvidas sobre APIs e ferramentas, consulte a documentação técnica no servidor interno."
                ],
                metadatas=[
                    {"source": "regras_negocio"},
                    {"source": "horario_operacao"},
                    {"source": "documentacao_interna"}
                ],
                ids=["doc1", "doc2", "doc3"]
            )
            print("Dados estáticos iniciais carregados.")