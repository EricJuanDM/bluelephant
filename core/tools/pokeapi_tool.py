import requests

def consultar_pokemon_data(nome: str) -> dict:
    nome = nome.lower().strip()
    url = f"https://pokeapi.co/api/v2/pokemon/{nome}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        
        info = {
            "id": data.get("id"),
            "nome": data.get("name").capitalize(),
            "tipos": [t['type']['name'].capitalize() for t in data.get("types", [])],
            "habilidades_principais": [
                a['ability']['name'].replace('-', ' ').capitalize() 
                for a in data.get("abilities", []) if not a['is_hidden']
            ][:2]
        }
        return {"status": "sucesso", "dados": info}

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {"status": "erro", "mensagem": f"Pokémon '{nome.capitalize()}' não encontrado."}
        return {"status": "erro", "mensagem": f"Erro HTTP ao acessar a PokéAPI: {e}"}
    
    except requests.exceptions.RequestException as e:
        return {"status": "erro", "mensagem": f"Erro de conexão com a PokéAPI: {e}"}

