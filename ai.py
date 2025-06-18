import os, sys, openai
from enum import Enum
from openai import OpenAI

class ModelProvider(str, Enum):
    OPENAI = "openai"
    QWEN = "qwen"

class ClientConfig:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs
    
    def generateClient(self) -> OpenAI:
        return OpenAI(*self.args, **self.kwargs)
    
    def __repr__(self):
        return f"ClientConfig(name={self.name}, args={self.args}, kwargs={self.kwargs})"
    
    @staticmethod
    def default() -> 'list[ClientConfig]':
        return [
            ClientConfig("openai"),
            ClientConfig(
                "qwen",
                api_key=os.environ["QWEN_API_KEY"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        ]

class LLMInterface:
    clients: dict[str, OpenAI] = {}
    disabled: bool = False
    
    @staticmethod
    def checkPermission():
        return "LLMINTERFACE_ENABLED" in os.environ and os.environ["LLMINTERFACE_ENABLED"] == "True"

    @staticmethod
    def initDefaultClients():
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."
        
        try:
            for clientConfig in ClientConfig.default():
                if clientConfig.name not in LLMInterface.clients:
                    LLMInterface.clients[clientConfig.name] = clientConfig.generateClient()
        except Exception as e:
            return "ERROR: Failed to initialise default clients; error: {}".format(e)
        
        return True
    
    @staticmethod
    def getClient(name: str) -> OpenAI | None:
        if not LLMInterface.checkPermission() or name not in LLMInterface.clients:
            return None
        
        return LLMInterface.clients[name]
    
    @staticmethod
    def addClient(config: ClientConfig) -> bool | str:
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."

        if config.name in LLMInterface.clients:
            return "ERROR: Client with name '{}' already exists.".format(config.name)
        
        try:
            LLMInterface.clients[config.name] = config.generateClient()
            return True
        except Exception as e:
            return "ERROR: Failed to add client '{}'; error: {}".format(config.name, e)
    
    @staticmethod
    def removeClient(name: str) -> bool | str:
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."
        
        if name not in LLMInterface.clients:
            return True
        
        try:
            del LLMInterface.clients[name]
            return True
        except Exception as e:
            return "ERROR: Failed to remove client '{}'; error: {}".format(name, e)
    
    # @staticmethod
    # def prompt(newPrompt, contextHistory=[], model="gpt-4o-mini", temperature=0.5, maxTokens=500):
    #     if not LLMInterface.initialised or LLMInterface.client == None:
    #         return "ERROR: Initialise LLMInterface first."
        
    #     sanitisedMessages = []
    #     if not isinstance(contextHistory, list):
    #         contextHistory = []
    #     else:
    #         sanitisedMessages = list(filter(lambda x: x != None, list(map(lambda message: {"role": message["role"], "content": message["content"]} if isinstance(message, dict) and "role" in message and "content" in message else None, contextHistory))))

    #     sanitisedMessages.append({
    #         "role": "user",
    #         "content": newPrompt
    #     })
        
    #     try:
    #         response = LLMInterface.client.chat.completions.create(
    #             model=model,
    #             messages=sanitisedMessages,
    #             temperature=temperature if isinstance(temperature, float) else 0.5,
    #             max_tokens=maxTokens if isinstance(maxTokens, int) else 500
    #         )
            
    #         return response.choices[0].message
    #     except Exception as e:
    #         return "ERROR: Failed to generate chat completion; error: {}".format(e)