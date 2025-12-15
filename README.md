#  Chatbot com Agente e Melhoria de Prompt em Tempo Real

Este projeto implementa um sistema de chatbot inteligente (LLM Agent) focado em **Melhoria Contínua de Prompt** através do feedback em tempo real do usuário. O agente utiliza o modelo Gemini, Vector Store (RAG) e integrações com APIs externas, tudo orquestrado via Python e Streamlit e empacotado com Docker.

##  Funcionalidades Principais

* **Agente Orquestrador:** Utiliza o LLM para tomar decisões sobre o uso de ferramentas e contexto.
* **Vector Store (ChromaDB):** Armazenamento de contexto interno (RAG) para respostas baseadas em dados estáticos.
* **Tools Integradas:** Capacidade de executar funções externas (ViaCEP e PokéAPI).
* **Feedback Inteligente:** O LLM processa o feedback do usuário e reescreve o prompt do sistema para melhorar o desempenho do agente em interações futuras (Aprendizado em Tempo Real).
* **Interface Streamlit:** Separação clara entre a área de Chat e a área de Feedback/Gerenciamento de Prompt.

##  Pré-requisitos

1.  **Docker Desktop:** Instalado e rodando (necessita de WSL 2 no Windows).
2.  **Chave de API:** Uma chave da Gemini API.
##  Como Rodar o Projeto (Usando Docker)

1.  **Clone o Repositório:**
    ```bash
    git clone [SEU LINK DO REPOSITÓRIO AQUI]
    cd nome-do-projeto
    ```

2.  **Configurar Variável de Ambiente:**
    * No arquivo `.env`, substitua `SUA_CHAVE_AQUI_OU_USE_DOTENV` pela sua chave da Gemini API:
        ```yaml
        # Exemplo:
         GEMINI_API_KEY=Sua_chave_real
        ```

3.  **Construir e Iniciar os Containers:**
    Execute o comando no diretório raiz do projeto:
    ```bash
    docker compose up --build -d
    ```

4.  **Acessar a Aplicação:**
    Após o Docker iniciar, acesse o seguinte endereço no seu navegador:
    ```
    http://localhost:8501
    ```
    Para parar e remover os containers:
    ```bash
    docker compose down
    ```

##  Estrutura do Projeto e Tecnologias

* **LLM:** Gemini API (gemini-2.5-flash)
* **Vector Store:** ChromaDB (Persistente via Volume Docker)
* **Framework Web:** Streamlit
* **Linguagem:** Python 3.11+

##  Exemplos de Uso (Demonstração de Tools e RAG)

Interaja com o Agente na aba "💬 Chat do Agente" com as seguintes perguntas:

| Pergunta | Esperado | Tool/Contexto Utilizado |
| :--- | :--- | :--- |
| **Qual o CEP de Curitiba?** | O agente deve usar a `ViaCEP` para fornecer o CEP de um endereço conhecido em Curitiba. | **ViaCEP API** |
| **Quem é o Pikachu e qual a principal habilidade dele?** | O agente deve usar a `PokéAPI` para buscar informações estruturadas sobre o Pokémon. | **PokéAPI** |
| **Qual é a política de devolução da empresa?** | O agente deve buscar o contexto na **ChromaDB** para responder à pergunta. | **Vector Store (RAG)** |

---

##  Documentação das APIs Utilizadas

### ViaCEP
* **Função:** Consulta de endereços a partir de um CEP.
* **Endpoint:** `https://viacep.com.br/ws/{CEP}/json/`
* **Arquivo:** `core/tools/viacep_tool.py`

### PokéAPI
* **Função:** Consulta de dados de Pokémon (tipo, habilidades, nome).
* **Endpoint:** `https://pokeapi.co/api/v2/pokemon/{NOME}`
* **Arquivo:** `core/tools/pokeapi_tool.py`
