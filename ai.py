import os, sys, openai
from enum import Enum
from openai import OpenAI

class LMProvider(str, Enum):
    OPENAI = "openai"
    QWEN = "qwen"

class LMVariants(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4_1_MINI = "gpt-4.1-mini"
    
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"
    
    QWEN_MAX = "qwen-max"
    QWEN_PLUS = "qwen-plus"
    QWEN_TURBO = "qwen-turbo"
    
    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"
    
    QWQ = "qwq-plus"
    QVQ = "qvq-max"
    
    QWEN3_8B = "qwen3-8b"
    QWEN3_14B = "qwen3-14b"
    QWEN3_32B = "qwen3-32b"
    QWEN3_30B_A3B = "qwen3-30b-a3b"
    QWEN3_235B_A22B = "qwen3-235b-a22b"

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

class Interaction:
    class Role(str, Enum):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        TOOL = "tool"
    
    def __init__(self, role: Role | str, content: str):
        if (not isinstance(role, str) and not isinstance(role, Interaction.Role)):
            raise ValueError("Role must be a string or an instance of Interaction.Role.")
        elif not isinstance(content, str):
            raise ValueError("Content must be a string.")
        
        self.role = role if isinstance(role, str) else role.value
        self.content = content
    
    def represent(self):
        return {
            "role": self.role,
            "content": self.content
        }

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
        if not LLMInterface.checkPermission():
            return None

        return LLMInterface.clients.get(name)
    
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
    
    @staticmethod
    def prompt(client: str, model: str, newPrompt: str, contextHistory: list[Interaction]=[], temperature=0.5, maxTokens=500):
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."
        
        client: OpenAI = LLMInterface.getClient(client)
        if client is None:
            return "ERROR: Client '{}' does not exist.".format(client)
        
        sanitisedMessages = []
        if not isinstance(contextHistory, list):
            contextHistory = []
        else:
            for item in contextHistory:
                if isinstance(item, Interaction):
                    sanitisedMessages.append(item.represent())

        sanitisedMessages.append({
            "role": Interaction.Role.USER.value,
            "content": newPrompt
        })
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=sanitisedMessages,
                temperature=temperature if isinstance(temperature, float) else 0.5,
                max_tokens=maxTokens if isinstance(maxTokens, int) else 500,
                extra_body={
                    "enable_thinking": False
                }
            )
            
            return response.choices[0].message
        except Exception as e:
            return "ERROR: Failed to generate chat completion; error: {}".format(e)