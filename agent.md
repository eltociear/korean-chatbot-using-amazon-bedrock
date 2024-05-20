# LangChain의 Agent 사용하기

[lambda-chat](./lambda-chat-ws/lambda_function.py)에서는 Agent를 정의하여 CoT를 구현합니다.

## 도서 정보 가져오기

교보문고의 Search API를 이용하여 아래와 같이 [도서정보를 가져오는 함수](https://colab.research.google.com/drive/1juAwGGOEiz7h3XPtCFeRyfDB9hspQdHc?usp=sharing)를 정의합니다.

```python
from langchain.agents import tool
import requests
from bs4 import BeautifulSoup

@tool
def get_product_list(keyword: str) -> list:
    """
    Search product list by keyword and then return product list
    keyword: search keyword
    return: product list
    """

    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        prod_list = [
            {"title": prod.text.strip(), "link": prod.get("href")} for prod in prod_info
        ]
        return prod_list[:5]
    else:
        return []
```

## Agent의 정의

아래와 같이 Agent를 ReAct로 정의합니다.

```python
from langchain.agents import AgentExecutor, create_react_agent

def use_agent(chat, query):
    tools = [check_system_time, get_product_list]
    prompt_template = get_react_prompt_template()
    print('prompt_template: ', prompt_template)
    
    agent = create_react_agent(chat, tools, prompt_template)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    response = agent_executor.invoke({"input": query})
    print('response: ', response)
    
    msg = response['output']
    print('msg: ', msg)
            
    return msg
```

이때, [ReAct를 위한 Prompt](https://github.com/chrishayuk/how-react-agents-work/blob/main/hello-agent-3.py#L20)는 아래와 같이 정의합니다.

```python
def get_react_prompt_template():
    # Get the react prompt template
    return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
```

