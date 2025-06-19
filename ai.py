import os, sys, openai, base64
from typing import List
from enum import Enum
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

class LMProvider(str, Enum):
    OPENAI = "openai"
    QWEN = "qwen"
    
    def __str__(self):
        return self.value

class LMVariant(str, Enum):
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
    
    def __str__(self):
        return self.value

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
    def default() -> 'List[ClientConfig]':
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

    def __init__(self, role: Role | str, content: str, imagePath: str | None=None, imageFileType: str | None=None, completionMessage: ChatCompletionMessage | None=None):
        if (not isinstance(role, str) and not isinstance(role, Interaction.Role)):
            raise ValueError("Role must be a string or an instance of Interaction.Role.")
        elif not isinstance(content, str):
            raise ValueError("Content must be a string.")
        elif imagePath is not None and not isinstance(imagePath, str):
            raise ValueError("Image path must be a string if provided.")
        elif imageFileType is not None and not isinstance(imageFileType, str):
            raise ValueError("Image file type must be a string if provided.")
        elif imagePath is not None and imageFileType is None:
            raise ValueError("Image file type must be provided if image path is given.")
        elif imagePath is not None and str(role) != Interaction.Role.USER.value:
            raise ValueError("Image input can only be included by the user role.")
        
        self.role: str = role if isinstance(role, str) else role.value
        self.content: str = content
        
        if imagePath != None:
            with open(imagePath, "rb") as f:
                data = f.read()
                self.imageData: str = base64.b64encode(data).decode("utf-8")
        else:
            self.imageData: str = None
        self.imageFileType: str = imageFileType
        self.completionMessage: ChatCompletionMessage | None = completionMessage

    def represent(self):
        if self.imageData is None:
            return {
                "role": self.role,
                "content": self.content
            }
        else:
            return {
                "role": self.role,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{self.imageFileType};base64,{self.imageData}"
                        }
                    },
                    {
                        "type": "text",
                        "text": self.content
                    }
                ]
            }
    
    def __str__(self):
        return f"""Interaction:
    Role: {self.role}
    Content: {self.content}
    Image Input: {'Yes' if self.imageData else 'No'}
    Image File Type: {self.imageFileType if self.imageFileType else 'None'}"""

class Tool:
    class Parameter:
        class Type(str, Enum):
            STRING = "string"
            INTEGER = "integer"
            NUMBER = "number"
            BOOLEAN = "boolean"
            ARRAY_NOT_RECOMMENDED = "array"
            OBJECT_NOT_RECOMMENDED = "object"

            def __str__(self):
                return self.value
        
        def __init__(self, name: str, dataType: Type, description: str, required: bool=False):
            self.name = name
            self.type = str(dataType)
            self.description = description
            self.required = required
        
    def __init__(self, callback, name: str, description: str, parameters: List[Parameter] | None=None):
        if not callable(callback):
            raise ValueError("Callback must be a callable function.")
        
        self.callback = callback
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def invoke(self, *args, **kwargs):
        return self.callback(*args, **kwargs)
    
    def represent(self) -> dict:
        data = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description
            }
        }
        
        if self.parameters is not None and isinstance(self.parameters, list) and len(self.parameters) > 0:
            data["function"]["parameters"] = {
                "type": "object",
                "properties": {param.name: {
                    "type": param.type,
                    "description": param.description
                } for param in self.parameters},
                "required": [param.name for param in self.parameters if param.required]
            }

        return data
        

class InteractionContext:
    def __init__(
        self,
        provider: LMProvider,
        variant: LMVariant,
        history: List[Interaction]=[],
        tools: List[Tool]=[],
        temperature: float=0.5,
        presence_penalty: float | None=None,
        top_p: float | None=None,
        top_k: int | None=None,
        preToolInvocationCallback=None,
        postToolInvocationCallback=None
    ):
        self.provider = provider
        self.variant = variant
        self.history = history
        self.tools = tools
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.preToolInvocationCallback = preToolInvocationCallback
        self.postToolInvocationCallback = postToolInvocationCallback
    
    def addInteraction(self, interaction: Interaction):
        self.history.append(interaction)

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
    def manualPrompt(client: str, *params) -> ChatCompletionMessage | str:
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."
        
        client: OpenAI = LLMInterface.getClient(client)
        if client is None:
            return "ERROR: Client '{}' does not exist.".format(client)
        
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message
        except Exception as e:
            return "ERROR: Failed to generate chat completion; error: {}".format(e)
    
    
    # @staticmethod
    # def prompt(client: str, model: str, newPrompt: str, contextHistory: list[Interaction]=[], temperature=0.5, maxTokens=500):
    #     if not LLMInterface.checkPermission():
    #         return "ERROR: LLMInterface does not have permission to operate."
        
    #     client: OpenAI = LLMInterface.getClient(client)
    #     if client is None:
    #         return "ERROR: Client '{}' does not exist.".format(client)
        
    #     sanitisedMessages = []
    #     if not isinstance(contextHistory, list):
    #         contextHistory = []
    #     else:
    #         for item in contextHistory:
    #             if isinstance(item, Interaction):
    #                 sanitisedMessages.append(item.represent())

    #     sanitisedMessages.append({
    #         "role": Interaction.Role.USER.value,
    #         "content": newPrompt
    #     })
        
    #     try:
    #         response = client.chat.completions.create(
    #             model=model,
    #             messages=sanitisedMessages,
    #             temperature=temperature if isinstance(temperature, float) else 0.5,
    #             max_tokens=maxTokens if isinstance(maxTokens, int) else 500
    #         )
            
    #         return response.choices[0].message
    #     except Exception as e:
    #         return "ERROR: Failed to generate chat completion; error: {}".format(e)