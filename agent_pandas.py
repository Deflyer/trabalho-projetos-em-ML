## Agent for Natural Language to Pandas Query (MÚLTIPLOS DATAFRAMES)

import os
from typing import Annotated, Sequence, TypedDict, Optional, Dict
import pandas as pd
import io 

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from operator import add as add_messages

# CONFIGURAÇÃO LLM

def build_llm(model: str = "x-ai/grok-4.1-fast:free", temperature: float = 0):
    """Constrói o modelo LLM com as configurações do OpenRouter."""
    llm = ChatOpenAI(
        model=model,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature
    )
    return llm

# VARIÁVEL GLOBAL DO DATAFRAME (Dicionário)
df_global_dict: Dict[str, pd.DataFrame] = {}


def get_dataframe_schema(
    df: pd.DataFrame,
    df_descricao: Optional[pd.DataFrame] = None,
    nome_df: str = "df_global",
    col_nome_campo: str = "Nome_do_campo",
    col_descricao: str = "descricao_do_campo"
) -> str:
    
    # 1. Preparar o mapeamento de descrições
    descricoes = {}
    if df_descricao is not None and not df_descricao.empty:
        # Cria um dicionário de mapeamento: Nome_do_campo -> descricao_do_campo
        try:
            # Garante que os nomes dos campos no mapeamento estejam em string para comparação
            descricoes = df_descricao.set_index(col_nome_campo)[col_descricao].astype(str).to_dict()
        except KeyError as e:
            # Caso os nomes das colunas de descrição não sejam encontrados
            print(f"Erro: As colunas '{col_nome_campo}' ou '{col_descricao}' não foram encontradas no df_descricao.")
            descricoes = {}

    # 2. Construir o esquema
    schema = []
    num_linhas = len(df)
    
    for col, dtype in df.dtypes.items():
        amostra_limite = 5
        if num_linhas > 0:
            sample = f"Exemplo: {list(df[col].head(amostra_limite).fillna('N/A'))}"
        else:
            sample = "Exemplo: [DataFrame vazio]"
        descricao = descricoes.get(col, '')
        
        schema_line = f"- Coluna **'{col}'**: Tipo **{dtype}**. "
        if descricao:
            schema_line += f"Descrição: **{descricao.strip()}**. "
        schema_line += sample
        
        schema.append(schema_line)

    return f"O DataFrame **'{nome_df}'** tem **{num_linhas}** linhas e as seguintes colunas:\n" + "\n".join(schema)


# FERRAMENTA PANDAS/PYTHON

@tool
def python_repl_tool(code: str) -> str:
    """Executa código Python no escopo que contém os DataFrames globais ('vendas', 'estoque', etc.) e retorna a saída (via print).
    O LLM DEVE ESPECIFICAR o nome do DataFrame a ser usado no código.
    """
    
    # 🆕 NOVO: Crie um dicionário para o ambiente de execução (locais)
    # Inclui todos os DFs do dicionário global e o pandas.
    execution_scope = {**df_global_dict, 'pd': pd}
    
    if not df_global_dict:
         return "Erro: Nenhum DataFrame disponível para execução."
    

    try:
        # Cria um buffer para capturar a saída (stdout) do print()
        old_stdout = os.sys.stdout
        os.sys.stdout = buffer = io.StringIO()
        
        # Executa o código usando o dicionário local (execution_scope)
        exec(code, execution_scope)
        
        # Captura e restaura o stdout
        os.sys.stdout = old_stdout 
        output = buffer.getvalue()
        
        # Retorna a saída capturada
        return f"Código executado com sucesso. Resultado:\n{output.strip() or 'Execução concluída sem saída de impressão. Use print() para obter resultados.'}"
        
    except Exception as e:
        return f"Erro na execução do código Python: {e}"


# BUILD DO AGENTE

class AgentState(TypedDict):
    """Define o estado do grafo: mensagens e o DataFrame (agora não usado aqui, mas mantido por compatibilidade)."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # ⚠️ df: Optional[pd.DataFrame] # Removido, pois a execução agora usa df_global_dict


def build_agent(llm, df_context: str):
    """
    Constrói o agente LangGraph Text-to-Pandas.
    df_context é o esquema completo de todos os DataFrames.
    """
    tools = [python_repl_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    # ... (função should_continue mantida) ...
    def should_continue(state: AgentState):
        """Roteador para decidir se deve continuar executando uma ferramenta ou terminar."""
        last = state["messages"][-1]
        if not hasattr(last, "tool_calls") or len(last.tool_calls) == 0:
            return END # Resposta final do LLM (Não chamou tool)
        
        tool_name = last.tool_calls[0].get("name") if isinstance(last.tool_calls[0], dict) else getattr(last.tool_calls[0], "name", None)
        return tool_name

    # 🆕 NOVO: System Prompt Adaptado
    system_prompt = (
        "Você é um assistente especialista em DataFrames Pandas. "
        "Sua função é traduzir perguntas em linguagem natural para código Python/Pandas para interagir com os DataFrames disponíveis. "
        "**PRIMEIRO, analise a pergunta e os esquemas para escolher o DataFrame mais relevante.** "
        "Após executar a ferramenta e analisar o resultado, **sempre inclua o código Python que foi rodado na sua resposta final**, formatando-o com um bloco de código Markdown (```python...```) logo antes da resposta em linguagem natural. "
        "Sempre use a ferramenta 'python_repl_tool' para executar o código necessário e obter a resposta. "
        "NÃO tente calcular a resposta diretamente. Gere o código Python/Pandas completo. "
        "O resultado do cálculo DEVE ser impresso (use 'print()') para que o resultado possa ser retornado. "
        "A seguir estão os esquemas de TODOS os DataFrames disponíveis:\n\n"
        f"{df_context}\n\n"
        "Exemplo de uso da ferramenta (usando o DF 'vendas'):\n"
        "Ação: `python_repl_tool(code=\"print(vendas['Valor'].sum())\")`\n"
        "REGRAS CRUCIAIS PARA O CÓDIGO PYTHON:"
        "1. SEMPRE use o **nome exato** de um dos DataFrames listados (ex: `vendas`, `estoque`) no seu código."
        "2. NUNCA use a variável genérica 'df_global'."
        "3. NUNCA, em hipótese alguma, crie ou recrie um DataFrame."
        "4. NUNCA importe bibliotecas."
        "5. SEMPRE garanta que o resultado final seja exibido usando 'print()'."
    )

    # ... (funções call_llm e take_action mantidas, mas take_action agora executa python_repl_tool) ...

    def call_llm(state: AgentState) -> AgentState:
        """Nó para invocar o LLM."""
        msgs = list(state["messages"])
        # Injeta o prompt do sistema no início de cada chamada ao LLM
        msgs = [SystemMessage(content=system_prompt)] + msgs 
        message = llm_with_tools.invoke(msgs)
        return {"messages": [message]}

    # O executor real não precisa de uma função auxiliar separada, pois a função
    # decorada '@tool python_repl_tool' foi adaptada para conter a lógica de 'exec()'
    # e buscar os DataFrames do dicionário global 'df_global_dict'.
    # O nó 'take_action' precisa apenas invocar a ferramenta.
    
    def take_action(state: AgentState) -> AgentState:
        """Nó para executar a ferramenta (agora o código) com o DF do estado."""
        tool_calls = state["messages"][-1].tool_calls
        results = []
        
        # ⚠️ Nota: A variável df_to_use não é mais passada para o exec,
        # pois o exec agora usa o dicionário global df_global_dict (ver python_repl_tool)
        
        for t in tool_calls:
            tool_name = t["name"]
            args = t["args"]
            
            if tool_name == python_repl_tool.name: 
                # LangGraph invoca a ferramenta aqui.
                # Como a ferramenta python_repl_tool usa o escopo global (df_global_dict),
                # a injeção acontece automaticamente dentro da função.
                tool_output = python_repl_tool.invoke(args) 
                
                # Para mostrar o código no resultado, fazemos um parse:
                args_code = args.get("code", "")

                # Combina o código e o output para passar ao LLM
                combined_content = (
                    f"CÓDIGO EXECUTADO:\n"
                    f"```python\n{args_code}\n```\n\n"
                    f"RESULTADO DA EXECUÇÃO:\n"
                    f"{tool_output}"
                )
                result = combined_content
            else:
                result = "Ferramenta incorreta ou desconhecida."
                
            results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result)))
                
        return {"messages": results}


    # Construção do Grafo LangGraph
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tool_executor", take_action) 

    # Roteamento: LLM -> Executor ou LLM -> Fim
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            python_repl_tool.name: "tool_executor",
            END: END,
        },
    )
    # Roteamento: Executor -> LLM (para formular a resposta final)
    graph.add_edge("tool_executor", "llm")
    graph.set_entry_point("llm")
    return graph.compile()