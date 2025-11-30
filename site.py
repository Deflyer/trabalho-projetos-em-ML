import os
import uuid
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain_core.messages import HumanMessage
from typing import Dict

# Importa as funções do agente (certifique-se de que agent_pandas.py está no mesmo diretório)
from agent_pandas import (
    build_llm,
    build_agent,
    df_global_dict as global_dfs_reference, # Importa a REFERÊNCIA GLOBAL (o dicionário)
    get_dataframe_schema,
)

# Constante para o nome do único usuário
SINGLE_USER = "Usuário" 

def ensure_session_state():
    """Inicializa todas as variáveis de sessão."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [] 
    if "agent" not in st.session_state:
        st.session_state.agent = None
        
    # DataFrames e Contextos
    if "dfs_data" not in st.session_state: 
        st.session_state.dfs_data: Dict[str, pd.DataFrame] = {} 
    if "dfs_context" not in st.session_state:
        st.session_state.dfs_context: Dict[str, str] = {} 

# Funções get_history_file_path e append_history_to_file (Omitidas para brevidade, mas devem existir no seu arquivo original)

def load_and_set_dataframes():
    """Cria múltiplos DataFrames de exemplo e configura os contextos."""
    
    # 1. ⚙️ DataFrames de Exemplo
    dfs = {
        "boletins": pd.read_csv("df_sibol.csv"),
        "envolvidos": pd.read_csv("df_sienv.csv"),
        "logradouros": pd.read_csv("df_silog.csv"),
        "veiculos": pd.read_csv("df_siveic.csv")
    }
    mapa_descricoes = {
        "boletins": pd.read_csv("dic_sibol.csv"),
        "envolvidos": pd.read_csv("dic_sienv.csv"),
        "logradouros": pd.read_csv("dic_silog.csv"),
        "veiculos": pd.read_csv("dic_siveic.csv")
    }

    # 2. Atualiza a referência global no módulo agent_pandas
    global global_dfs_reference
    global_dfs_reference.update(dfs) 

    # 3. Armazena no Session State e gera os contextos
    st.session_state.dfs_data = dfs
    
    contexts = {}

    for nome_dados, df_dados in dfs.items():        
        df_dic = mapa_descricoes[nome_dados]
        
        contexts[nome_dados] = get_dataframe_schema(
            df=df_dados,
            df_descricao=df_dic,
            nome_df=nome_dados,
            col_nome_campo="Nome_do_campo",
            col_descricao="descricao_do_campo"
        )
    
    st.session_state.dfs_context = contexts
    st.success(f"Carregados {len(dfs)} DataFrames: {', '.join(dfs.keys())}.")


def main():
    # Assegura que as variáveis de ambiente e o estado da sessão estão prontos
    load_dotenv()
    st.set_page_config(page_title="Pandas Chat (aula)", page_icon="📈")
    ensure_session_state()

    st.title("📈 Agente interativo com os dados de trânsito")
    st.caption("Faça perguntas, e o agente escolherá o DataFrame adequado e gerará a resposta")

    # Inicializa os DataFrames na sessão
    if not st.session_state.dfs_data:
        load_and_set_dataframes()

    with st.sidebar:
        st.header("📊 DataFrames disponíveis")
        
        all_dfs = st.session_state.dfs_data
        all_contexts = st.session_state.dfs_context

        if all_dfs:
            st.markdown(f"**Total de DFs:** `{len(all_dfs)}`")
            
            # Mostra o Dataframe
            with st.expander("Ver Schemas e Dados"):
                full_context_text = ""
                for name, context in all_contexts.items():
                    st.markdown(f"**DataFrame: `{name}`**") 
                    full_context_text += f"**DataFrame: `{name}`**\n"
                    full_context_text += context + "\n\n"
                    st.dataframe(all_dfs[name].head())
                
                # Armazena o texto completo para o Agente usar
                st.session_state.full_df_context = full_context_text
        else:
            st.warning("Nenhum DataFrame carregado.")

        st.header("🤖 Agent")

        # Botão para criação do agente
        if st.button("(re)Create Agent", type="primary"):
            if not st.session_state.dfs_data:
                st.warning("Por favor, carregue os DataFrames primeiro.")
            else:
                with st.spinner("Criando agente..."):
                    llm = build_llm(temperature=0)
                    df_context = st.session_state.full_df_context 
                    st.session_state.agent = build_agent(llm, df_context) 
                st.success("Agente pronto para uso!")
        
        if st.session_state.agent:
            st.success("Agente ativo.")
        else:
            st.info("Crie o agente para começar a fazer perguntas.")


    # --- Área de Chat ---
    st.subheader("Bate-papo")
    
    # Itera sobre todas as mensagens armazenadas no estado da sessão.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Usa o prefixo manual que você definiu para o usuário para consistência
            if message["role"] == "user":
                st.markdown(f"**{SINGLE_USER}**: {message['content']}")
            else:
                st.markdown(message["content"])


    if prompt := st.chat_input(f"{SINGLE_USER} pergunta: "):
        # Adiciona a mensagem do usuário ao estado
        user_msg = {"role": "user", "content": prompt} 
        st.session_state.messages.append(user_msg)
        
        # Exibe a mensagem do usuário imediatamente
        with st.chat_message("user"):
            st.markdown(f"**{SINGLE_USER}**: {prompt}") 
        
        if st.session_state.agent is None:
            with st.chat_message("assistant"):
                st.warning("Crie o agente na barra lateral para fazer perguntas sobre os dados.")
        else:
            clean_prompt = prompt.replace("@colaborai", "").strip()
            
            # Executa o agente e exibe a resposta
            with st.chat_message("assistant"):
                with st.spinner("Pensando e executando código Pandas..."):
                    result = st.session_state.agent.invoke({
                        "messages": [HumanMessage(content=clean_prompt)],
                    })
                    
                    content = result["messages"][-1].content
                    st.markdown(content)
                    
                    # Adiciona a mensagem do assistente ao estado (após a exibição)
                    assistant_msg = {"role": "assistant", "content": content} 
                    st.session_state.messages.append(assistant_msg)

if __name__ == "__main__":
    main()