import requests

def consultar_cep(cep: str) -> dict:
    cep = ''.join(filter(str.isdigit, cep))
    url = f"https://viacep.com.br/ws/{cep}/json/"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        data = response.json()
        
        if 'erro' in data:
            return {"status": "erro", "mensagem": "CEP não encontrado."}
        
        return data

    except requests.exceptions.RequestException as e:
        return {"status": "erro", "mensagem": f"Erro ao acessar a ViaCEP: {e}"}

