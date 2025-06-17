import os, sys, openai
from openai import OpenAI

class OpenAIChat:
    client: OpenAI = None
    initialised = False
    
    @staticmethod
    def checkPermission():
        return "OPENAICHAT_ENABLED" in os.environ and os.environ["OPENAICHAT_ENABLED"] == "True"
    
    @staticmethod
    def initialise():
        if not OpenAIChat.checkPermission():
            return "ERROR: OpenAIChat does not have permission to operate."
        if "OPENAI_API_KEY" not in os.environ:
            return "ERROR: OPENAI_API_KEY is not set."
        
        try:
            OpenAIChat.client = OpenAI()
        except Exception as e:
            return "ERROR: Failed to initialise OpenAI client; error: {}".format(e)
        
        OpenAIChat.initialised = True
        return True
    
    @staticmethod
    def prompt(newPrompt, contextHistory=[], model="gpt-4o-mini", temperature=0.5, maxTokens=500):
        if not OpenAIChat.initialised or OpenAIChat.client == None:
            return "ERROR: Initialise OpenAIChat first."
        
        sanitisedMessages = []
        if not isinstance(contextHistory, list):
            contextHistory = []
        else:
            sanitisedMessages = list(filter(lambda x: x != None, list(map(lambda message: {"role": message["role"], "content": message["content"]} if isinstance(message, dict) and "role" in message and "content" in message else None, contextHistory))))
            
        sanitisedMessages.append({
            "role": "user",
            "content": newPrompt
        })
        
        try:
            response = OpenAIChat.client.chat.completions.create(
                model=model,
                messages=sanitisedMessages,
                temperature=temperature if isinstance(temperature, float) else 0.5,
                max_tokens=maxTokens if isinstance(maxTokens, int) else 500
            )
            
            return response.choices[0].message
        except Exception as e:
            return "ERROR: Failed to generate chat completion; error: {}".format(e)