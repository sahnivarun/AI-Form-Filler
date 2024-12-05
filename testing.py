from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import ChatPromptTemplate
class AIModel(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass
class OpenAIModel(AIModel):
    def __init__(self, api_key: str, llm_model: str):
        from langchain_openai import ChatOpenAI

        self.model = ChatOpenAI(
            model_name=llm_model, openai_api_key=api_key, temperature=0.4
        )

    def invoke(self, prompt: str) -> BaseMessage:
        response = self.model.invoke(prompt)
        return response
api_key = "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"
llm_model = "gpt-4o-mini"
model = OpenAIModel(api_key, llm_model)
prompt = "how are you?"
response = model.invoke(prompt)
print(response)