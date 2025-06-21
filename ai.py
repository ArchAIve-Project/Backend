import os, sys, openai, base64, json
from typing import List
from enum import Enum
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall

class LMProvider(str, Enum):
    """
    An enumeration representing supported Language Model (LM) providers.
    Attributes:
        OPENAI: Represents the OpenAI language model provider.
        QWEN: Represents the Qwen language model provider.
    Methods:
        __str__(): Returns the string value of the LMProvider instance.
    """
    OPENAI = "openai"
    QWEN = "qwen"
    
    def __str__(self):
        return self.value

class LMVariant(str, Enum):
    """
    # Enumeration of Supported Language Model (LM) Variants

    Each member represents a specific LM variant, identified by its string value.

    ## Variants

    - **GPT_4O**: `"gpt-4o"`  
      GPT-4 Omni model.
    - **GPT_4O_MINI**: `"gpt-4o-mini"`  
      Mini version of GPT-4 Omni.
    - **GPT_4_1**: `"gpt-4.1"`  
      GPT-4.1 model.
    - **GPT_4_1_NANO**: `"gpt-4.1-nano"`  
      Nano version of GPT-4.1.
    - **GPT_4_1_MINI**: `"gpt-4.1-mini"`  
      Mini version of GPT-4.1.
    
    
    - **O3**: `"o3"`  
      O3 model.
    - **O3_MINI**: `"o3-mini"`  
      Mini version of O3.
    - **O4_MINI**: `"o4-mini"`  
      Mini version of O4.
    
    
    - **QWEN_MAX**: `"qwen-max"`  
      Qwen Max model.
    - **QWEN_PLUS**: `"qwen-plus"`  
      Qwen Plus model.
    - **QWEN_TURBO**: `"qwen-turbo"`  
      Qwen Turbo model.
    
    
    - **QWEN_VL_MAX**: `"qwen-vl-max"`  
      Qwen Vision-Language Max model.
    - **QWEN_VL_PLUS**: `"qwen-vl-plus"`  
      Qwen Vision-Language Plus model.
    
    
    - **QWQ**: `"qwq-plus"`  
      QWQ Plus model.
    - **QVQ**: `"qvq-max"`  
      QVQ Max model.
    
    
    - **QWEN3_8B**: `"qwen3-8b"`  
      Qwen3 8B parameter model.
    - **QWEN3_14B**: `"qwen3-14b"`  
      Qwen3 14B parameter model.
    - **QWEN3_32B**: `"qwen3-32b"`  
      Qwen3 32B parameter model.
    - **QWEN3_30B_A3B**: `"qwen3-30b-a3b"`  
      Qwen3 30B A3B variant.
    - **QWEN3_235B_A22B**: `"qwen3-235b-a22b"`  
      Qwen3 235B A22B variant.

    The `__str__` method returns the string value of the variant.
    """
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
    """
    ClientConfig is a configuration class for initializing and managing client instances for different AI service providers.
    Attributes:
        name (str): The name of the client or service provider.
        args (tuple): Positional arguments to be passed to the client constructor.
        kwargs (dict): Keyword arguments to be passed to the client constructor.
    Methods:
        generateClient() -> OpenAI:
            Instantiates and returns an OpenAI client using the stored arguments.
        __repr__():
            Returns a string representation of the ClientConfig instance.
        default() -> List[ClientConfig]:
            Returns a list of default ClientConfig instances for supported providers.
    """
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
    """
    Represents an interaction in a conversational AI system, supporting user, assistant, system, and tool roles, 
    as well as optional image and tool call data.
    
    ### Classes
        Role (Enum): Defines valid roles for an interaction ('user', 'assistant', 'system', 'tool').
    ### Args
        role (Role | str): The role of the interaction. Must be one of the defined roles or a string.
        content (str | None): The textual content of the interaction. Can be None.
        imagePath (str | None, optional): Path to an image file to include in the interaction. Only allowed for 'user' role.
        imageFileType (str | None, optional): MIME type of the image file. Required if imagePath is provided.
        toolCallID (str | None, optional): Identifier for a tool call. Only allowed for 'tool' role.
        toolCallName (str | None, optional): Name of the tool being called. Required if toolCallID is provided.
        completionMessage (ChatCompletionMessage | None, optional): Associated completion message, if any.
    ### Raises
        ValueError: If any argument does not meet the required type or logical constraints.
    
    ### Attributes
        role (str): The role of the interaction.
        content (str | None): The textual content of the interaction.
        imageData (str | None): Base64-encoded image data, if an image is included.
        imageFileType (str | None): MIME type of the image, if provided.
        toolCallID (str | None): Tool call identifier, if provided.
        toolCallName (str | None): Tool call name, if provided.
        completionMessage (ChatCompletionMessage | None): Associated completion message, if any.
    
    ### Methods:
        represent(): Returns a dictionary representation of the interaction, formatted for downstream processing.
        __str__(): Returns a string summary of the interaction instance.
    """
    class Role(str, Enum):
        """
        An enumeration representing the different roles in a conversational AI context.
        Attributes:
            USER (str): Represents the user interacting with the AI.
            ASSISTANT (str): Represents the AI assistant.
            SYSTEM (str): Represents system-level messages or instructions.
            TOOL (str): Represents a tool or function used within the conversation.
        Methods:
            __str__(): Returns the string value of the role.
        """
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        TOOL = "tool"
        
        def __str__(self):
            return self.value

    def __init__(self, role: Role | str, content: str | None, imagePath: str | None=None, imageFileType: str | None=None, toolCallID: str | None=None, toolCallName: str | None=None, completionMessage: ChatCompletionMessage | None=None):
        if (not isinstance(role, str) and not isinstance(role, Interaction.Role)):
            raise ValueError("Role must be a string or an instance of Interaction.Role.")
        elif content is not None and not isinstance(content, str):
            raise ValueError("Content must be a string or None.")
        elif imagePath is not None and not isinstance(imagePath, str):
            raise ValueError("Image path must be a string if provided.")
        elif imageFileType is not None and not isinstance(imageFileType, str):
            raise ValueError("Image file type must be a string if provided.")
        elif imagePath is not None and imageFileType is None:
            raise ValueError("Image file type must be provided if image path is given.")
        elif imagePath is not None and str(role) != Interaction.Role.USER.value:
            raise ValueError("Image input can only be included by the user role.")
        elif toolCallID is not None and not isinstance(toolCallID, str):
            raise ValueError("Tool call ID must be a string if provided.")
        elif toolCallName is not None and not isinstance(toolCallName, str):
            raise ValueError("Tool call name must be a string if provided.")
        elif toolCallID is not None and (toolCallName is None or content is None):
            raise ValueError("Tool call name and content must be provided if tool call ID is given.")
        elif toolCallID is not None and str(role) != Interaction.Role.TOOL.value:
            raise ValueError("Tool call can only be included by the tool role.")
        
        self.role: str = str(role)
        self.content: str | None = content

        if imagePath != None:
            with open(imagePath, "rb") as f:
                data = f.read()
                self.imageData: str = base64.b64encode(data).decode("utf-8")
        else:
            self.imageData: str = None
        self.imageFileType: str = imageFileType
        self.toolCallID: str | None = toolCallID
        self.toolCallName: str | None = toolCallName
        self.completionMessage: ChatCompletionMessage | None = completionMessage

    def represent(self):
        """
        Generates a dictionary representation of this interaction suitable for inclusion in the
        'messages' parameter of the chat completions API.

        - For 'tool' role: returns a dict with tool_call_id, name, and content.
        - For user role with image: returns a dict with content as a list containing image_url and text.
        - For all others: returns a dict with role and content.

        Returns:
            dict: Dictionary formatted for chat completions API.
        """
        if self.role == Interaction.Role.TOOL.value:
            return {
                "role": self.role,
                "tool_call_id": self.toolCallID,
                "name": self.toolCallName,
                "content": self.content
            }
        elif self.imageData is not None:
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
        else:
            return {
                "role": self.role,
                "content": [
                    {
                        "type": "text",
                        "text": self.content
                    }
                ]
            }
    
    def __str__(self):
        return f"Interaction(role={self.role}, content={self.content}, imageDataLength={len(self.imageData) if self.imageData else None}, imageFileType={self.imageFileType}, toolCallID={self.toolCallID}, toolCallName={self.toolCallName}, completionMessageExists={self.completionMessage is not None})"

class Tool:
    """
    A class representing a callable tool with metadata and parameter specification.
    Classes:
        Parameter:
            Represents a parameter for the tool, including its name, type, description, and whether it is required.
            Classes:
                Type (Enum):
                    Enum for supported parameter types: STRING, INTEGER, NUMBER, BOOLEAN, ARRAY_NOT_RECOMMENDED, OBJECT_NOT_RECOMMENDED.
    Attributes:
        callback (callable): The function to be invoked by the tool.
        name (str): The name of the tool.
        description (str): A description of the tool.
        parameters (List[Parameter] | None): Optional list of Parameter objects describing the tool's parameters.
    Methods:
        __init__(callback, name, description, parameters=None):
            Initializes the Tool with a callback, name, description, and optional parameters.
        invoke(*args, **kwargs):
            Invokes the tool's callback with the provided arguments.
        represent() -> dict:
            Returns a dictionary representation of the tool, suitable for serialization or API documentation.
        __str__():
            Returns a string representation of the Tool instance.
    
    Contains a nested class `Parameter` to define the parameters of the tool.
    """
    class Parameter:
        """
        Represents a parameter with a name, type, description, and required flag.
        
        Args:
            name (str): The name of the parameter.
            dataType (Type): The data type of the parameter, as a member of the Type enum.
            description (str): A description of the parameter.
            required (bool, optional): Whether the parameter is required. Defaults to False.
        Methods:
            __str__(): Returns a string representation of the Parameter instance.
        
        Contains a nested enum `Type` to define the supported data types for parameters.
        """
        class Type(str, Enum):
            """
            An enumeration representing supported data types.

            Attributes:
                STRING (str): Represents a string data type.
                INTEGER (str): Represents an integer data type.
                NUMBER (str): Represents a numeric data type (integer or float).
                BOOLEAN (str): Represents a boolean data type.
                ARRAY_NOT_RECOMMENDED (str): Represents an array data type (not recommended).
                OBJECT_NOT_RECOMMENDED (str): Represents an object data type (not recommended).

            Methods:
                __str__(): Returns the string value of the enum member.
            """
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
        
        def __str__(self):
            return f"Parameter(name={self.name}, type={self.type}, description={self.description}, required={self.required})"
        
    def __init__(self, callback, name: str, description: str, parameters: List[Parameter] | None=None):
        if not callable(callback):
            raise ValueError("Callback must be a callable function.")
        
        self.callback = callback
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def invoke(self, *args, **kwargs):
        """
        Invokes the stored callback function with the provided arguments.

        Args:
            *args: Variable length argument list to pass to the callback.
            **kwargs: Arbitrary keyword arguments to pass to the callback.

        Returns the result returned by the callback function.
        """
        return self.callback(*args, **kwargs)
    
    def represent(self) -> dict:
        """
        Returns a dictionary representation of the function, including its name, description, and optionally its parameters.
        The returned dictionary follows a specific structure:
        - "type": Always set to "function".
        - "function": Contains:
            - "name": The name of the function.
            - "description": The description of the function.
            - "parameters" (optional): If parameters are defined and non-empty, includes:
                - "type": Always set to "object".
                - "properties": A dictionary mapping parameter names to their type and description.
                - "required": A list of parameter names that are required.
        Returns:
            dict: The structured representation of the function and its parameters.
        """
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

    def __str__(self):
        return f"Tool(name={self.name}, description={self.description}, parameters={', '.join(str(param) for param in self.parameters) if self.parameters else 'None'})"
        

class InteractionContext:
    """
    InteractionContext manages the context and configuration for a sequence of interactions with a language model (LM), including conversation history, model parameters, and tool integrations.
    Attributes:
        provider (LMProvider): The language model provider.
        variant (LMVariant): The specific model variant to use.
        history (List[Interaction]): List of past interactions in the conversation.
        tools (List[Tool] | None): Optional list of tools available for tool-augmented interactions.
        temperature (float | None): Sampling temperature for model generation.
        presence_penalty (float | None): Penalty for repeated tokens in model output.
        top_p (float | None): Nucleus sampling parameter for model generation.
        preToolInvocationCallback (callable | None): Optional callback before tool invocation.
        postToolInvocationCallback (callable | None): Optional callback after tool invocation.
    Methods:
        addInteraction(interaction: Interaction, imageMessageAcknowledged: bool=False):
            Adds an interaction to the history, with validation for image data and model capability.
        generateInputMessages() -> List[dict]:
            Generates a list of message dictionaries representing the conversation history, formatted for model input.
        promptKwargs() -> dict:
            Assembles and returns a dictionary of parameters for prompting the language model, including model, messages, tools, and generation settings.
        __str__():
            Returns a human-readable string representation of the InteractionContext instance, including provider, variant, history, tools, and configuration.
    
    ### Usage Guide
    - Main variations of `InteractionContext` that are typically employed:
        - Regular prompting: Normal and VL models are fine. Tool invocations can be used.
        - Image understanding: VL models are required. Ensure `imageMessageAcknowledged` is set to True when adding interactions. VL models cannot invoke tools.
        - Tool invocations: Can be used with any model variant, but ensure the model supports tool calls.
    - Do `print(cont)` to get a comprehensive overview of the entire context quickly. Helpful for debugging purposes.
    - Do not use `temperature` and `top_p` at the same time.
    - Avoid using `presence_penalty` and `top_p` at the same time, as they may conflict in some models.
    - Returning `False` from a `preToolInvocationCallback` or `postToolInvocationCallback` will interrupt the `LLMInterface.engage` process and return immediately.
    - The `addInteraction` method raises an exception if an interaction with image data is added without acknowledging that the model variant supports image understanding.
    - This class should be used to encapsulate the context of a conversation with a language model, allowing for structured interactions and tool invocations.
    
    Advanced:
    `Interaction` and `Tool` classes are Pythonic representations of JSON schemas required by the ultimate underlying Chat Completions API.
    It is possible to use these classes to manually construct interactions and tools, and setup a custom workflow with `LLMInterface.manualPrompt`.
    """
    def __init__(
        self,
        provider: LMProvider,
        variant: LMVariant,
        history: List[Interaction]=[],
        tools: List[Tool] | None=None,
        temperature: float | None=None,
        presence_penalty: float | None=None,
        top_p: float | None=None,
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
        self.preToolInvocationCallback = preToolInvocationCallback
        self.postToolInvocationCallback = postToolInvocationCallback
    
    def addInteraction(self, interaction: Interaction, imageMessageAcknowledged: bool=False):
        """Adds an interaction to the context.

        Args:
            interaction (Interaction): The interaction to add.
            imageMessageAcknowledged (bool, optional): Whether the image message has been acknowledged. Defaults to False.

        Raises:
            Exception: If the interaction contains image data and the model variant does not support it.
        """
        if interaction.imageData is not None and not imageMessageAcknowledged:
            raise Exception("Interaction with image data requires a model variant capable of image understanding. Silence this by ensuring you have an appropriate model variant set and by setting imageMessageAcknowledged to True.")
        self.history.append(interaction)
    
    def generateInputMessages(self) -> List[dict]:
        """
        Generates a list of message dictionaries representing the conversation history.
        Iterates through each interaction in the history. If an interaction contains a completion message with tool calls,
        it appends the direct JSON representation of the completion message. Otherwise, it appends a custom representation
        of the interaction (for user, assistant, system, or tool call messages).
        Returns:
            List[dict]: A list of message dictionaries formatted for input to a language model or API.
        """
        messages = []
        for interaction in self.history:
            if interaction.completionMessage and interaction.completionMessage.tool_calls:
                # Add tool call *request* messages in direct JSON representation
                messages.append(interaction.completionMessage.to_dict())
            else:
                # Add regular user/assistant/system/tool *call* messages in custom representation format
                messages.append(interaction.represent())
        
        return messages
    
    def promptKwargs(self) -> dict:
        """
        Generates a dictionary of keyword arguments for prompting an AI model.
        Returns:
            dict: A dictionary containing the model variant, input messages, and optionally
            tools, temperature, presence penalty, and top_p parameters if they are set.
        The returned dictionary includes:
            - "model": The model variant as a string.
            - "messages": The input messages generated by `generateInputMessages()`.
            - "tools": A list of tool representations if `self.tools` is set.
            - "temperature": The temperature value if it is a float.
            - "presence_penalty": The presence penalty value if it is a float.
            - "top_p": The top_p value if it is a float.
        """
        params = {
            "model": str(self.variant),
            "messages": self.generateInputMessages()
        }
        
        if self.tools:
            params["tools"] = [tool.represent() for tool in self.tools]
        if isinstance(self.temperature, float):
            params["temperature"] = self.temperature
        if isinstance(self.presence_penalty, float):
            params["presence_penalty"] = self.presence_penalty
        if isinstance(self.top_p, float):
            params["top_p"] = self.top_p
        
        return params
    
    def __str__(self):        
        return f"""<InteractionContext Instance:
Provider: {self.provider}
Variant: {self.variant}
History:
---
{"\n".join(str(interaction) for interaction in self.history) if self.history else "None"}
---
Tools:
---
{"\n".join(str(tool) for tool in self.tools) if self.tools else "None"}
---
Temperature: {self.temperature}
Presence Penalty: {self.presence_penalty}
Top P: {self.top_p}
Pre-Tool Invocation Callback: {"Yes" if self.preToolInvocationCallback else "No"}
Post-Tool Invocation Callback: {"Yes" if self.postToolInvocationCallback else "No"} />"""

class LLMInterface:
    """
    LLMInterface provides a static interface for managing and interacting with multiple Chat Completions API supported LLM clients.
    Attributes:
        clients (dict[str, OpenAI]): A dictionary mapping client names to OpenAI client instances.
        disabled (bool): A flag indicating whether the interface is disabled.
    Methods:
        checkPermission() -> bool:
            Checks if the LLMInterface is enabled via the "LLMINTERFACE_ENABLED" environment variable.
        initDefaultClients() -> bool | str:
            Initializes default OpenAI clients using configurations from ClientConfig.default().
            Returns True on success, or an error message string on failure or lack of permission.
        getClient(name: str) -> OpenAI | None:
            Retrieves an OpenAI client by name if permission is granted, otherwise returns None.
        addClient(config: ClientConfig) -> bool | str:
            Adds a new OpenAI client using the provided ClientConfig.
            Returns True on success, or an error message string on failure or lack of permission.
        removeClient(name: str) -> bool | str:
            Removes an OpenAI client by name.
            Returns True on success, or an error message string on failure or lack of permission.
        manualPrompt(client: str, **params) -> ChatCompletionMessage | str:
            Sends a manual prompt to the specified client with given parameters.
            Returns the resulting ChatCompletionMessage, or an error message string on failure or lack of permission.
        engage(context: InteractionContext) -> ChatCompletionMessage | str:
            Engages in a multi-turn interaction with a client, handling tool calls and callbacks as specified in the context.
            Returns the final ChatCompletionMessage, or an error message string on failure or lack of permission.
    
    ----
    
    ### Usage Guide
    - You need a valid `ClientConfig` to add a new client. Parameters passed to `ClientConfig` should match the constructor of the `OpenAI` client.
    - Ensure that the environment variable `LLMINTERFACE_ENABLED` is set to "True" to allow operations.
    - Use `LLMInterface.initDefaultClients()` to initialize default clients based on the configurations in `ClientConfig.default()`.
    - After setting up and adding a client, you can use it simply by referencing its name (`ClientConfig.name`).
    - Use `LLMInterface.manualPrompt(client: str, **params)` to send a prompt through a specific client manually.
    - Use `LLMInterface.engage(context: InteractionContext)` to engage LLMs in a complex interaction, including tool invocations and callbacks. Relies on `InteractionContext` to upkeep the conversation state and tools.
    
    Sample code (Non-`InteractionContext` usage):
    ```python
    from openai.types.chat import ChatCompletionMessage
    from ai import LLMInterface, ClientConfig
    
    openaiConfig = ClientConfig(
        "openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1"
    )
    
    LLMInterface.addClient(openaiConfig)
    
    message: ChatCompletionMessage = LLMInterface.manualPrompt(
        "openai",
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello, who are you?"}
        ]
    )
    print(message.content)  # Outputs the response from the LLM
    ```
    
    Sample Code (Using `InteractionContext`):
    ```python
    from openai.types.chat import ChatCompletionMessage
    from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant, ClientConfig, Tool

    LLMInterface.initDefaultClients() # includes a "openai" (LMProvider.OPENAI) and "qwen" (LMProvider.QWEN) client as of writing

    def get_BMI(weight: float, height: float) -> str:
        return str(weight / (height ** 2))

    bmiTool = Tool(
        callback=get_BMI,
        name="get_BMI",
        description="Calculate Body Mass Index (BMI) from weight (kg) and height (m) inputs.",
        parameters=[
            Tool.Parameter(
                name="weight",
                description="Weight in kilograms (kg).",
                dataType=Tool.Parameter.Type.NUMBER,
                required=True
            ),
            Tool.Parameter(
                name="height",
                description="Height in meters (m).",
                dataType=Tool.Parameter.Type.NUMBER,
                required=True
            )
        ]
    )

    cont = InteractionContext(
        provider=LMProvider.OPENAI,
        variant=LMVariant.GPT_4O_MINI,
        history=[],
        tools=[bmiTool],
        temperature=0.8,
        preToolInvocationCallback=lambda toolInvocMsg: print("Tool invocation message:", toolInvocMsg),
        postToolInvocationCallback=lambda toolResultMsg: print("Tool invocation result:", toolResultMsg)
    )

    cont.addInteraction(
        Interaction(
            role=Interaction.Role.SYSTEM,
            content="You are a helpful assistant that can calculate BMI using the get_BMI tool."
        )
    )

    cont.addInteraction(
        Interaction(
            role=Interaction.Role.USER,
            content="What is my BMI if I weigh 70 kg and am 1.75 m tall?"
        )
    )

    response: ChatCompletionMessage = LLMInterface.engage(cont)
    if not isinstance(response, str):
        print("Response from LLM:", response.content) # Your BMI is approximately 22.86
    else:
        print("Error response:", response)
    ```
    
    Note: Ensure that environment variables and other requirements are set up correctly for clients to function properly.
    """
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
    def manualPrompt(client: str, **params) -> ChatCompletionMessage | str:
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."
        
        client: OpenAI = LLMInterface.getClient(client)
        if client is None:
            return "ERROR: Client '{}' does not exist.".format(client)
        
        try:
            response: ChatCompletion = client.chat.completions.create(**params)
            return response.choices[0].message
        except Exception as e:
            return "ERROR: Failed to generate chat completion; error: {}".format(e)
    
    @staticmethod
    def engage(context: InteractionContext) -> ChatCompletionMessage | str:
        if not LLMInterface.checkPermission():
            return "ERROR: LLMInterface does not have permission to operate."
        
        client: OpenAI = LLMInterface.getClient(context.provider.value)
        if client is None:
            return "ERROR: Client '{}' does not exist.".format(context.provider.value)
        
        # Initial prompt execution
        response: ChatCompletion | None = None
        try:
            response: ChatCompletion = client.chat.completions.create(**context.promptKwargs())
        except Exception as e:
            return "ERROR: Failed to generate chat completion; error: {}".format(e)
        
        context.addInteraction(
            Interaction(
                role=Interaction.Role.ASSISTANT,
                content=response.choices[0].message.content,
                completionMessage=response.choices[0].message
            )
        )
        
        while response.choices[0].message.tool_calls:
            # Identify tool call request
            try:
                if context.preToolInvocationCallback is not None:
                    if context.preToolInvocationCallback(response.choices[0].message) == False:
                        return "ERROR: Pre-tool invocation callback returned False, stopping execution."
                
                toolCall: ChatCompletionMessageToolCall = response.choices[0].message.tool_calls[0]
                tool: Tool = None
                if context.tools is None or len(context.tools) == 0:
                    raise Exception("Invocation request for tool '{}' but no tools are available in context.".format(toolCall.function.name))
                for t in context.tools:
                    if t.name == toolCall.function.name:
                        tool = t
                        break
                if tool is None:
                    return "ERROR: Tool '{}' invoked but not found in context tools.".format(toolCall.function.name)
            except Exception as e:
                return "ERROR: Failed to begin tool invocation; error: {}".format(e)
            
            # Carry out tool invocation
            try:
                func_args = toolCall.function.arguments
                func_args = json.loads(func_args)
                out = str(tool.invoke(**func_args))
                
                context.addInteraction(
                    Interaction(
                        role=Interaction.Role.TOOL,
                        content=out,
                        toolCallID=toolCall.id,
                        toolCallName=toolCall.function.name
                    )
                )
            except Exception as e:
                return "ERROR: Failed to invoke tool '{}'; error: {}".format(toolCall.function.name, e)
            
            # Get response after tool invocation
            try:
                response = client.chat.completions.create(**context.promptKwargs())
            except Exception as e:
                return "ERROR: Failed to generate post-tool chat completion; error: {}".format(e)
            
            context.addInteraction(
                Interaction(
                    role=Interaction.Role.ASSISTANT,
                    content=response.choices[0].message.content,
                    completionMessage=response.choices[0].message
                )
            )
            
            if context.postToolInvocationCallback is not None:
                try:
                    if context.postToolInvocationCallback(response.choices[0].message) == False:
                        return "ERROR: Post-tool invocation callback returned False, stopping execution."
                except Exception as e:
                    return "ERROR: Failed to execute post-tool callback; error: {}".format(e)
        
        return response.choices[0].message