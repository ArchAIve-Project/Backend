from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant
from schemas import Artefact
import sys

class InItemChatbot:
    """
    A chatbot for generating conversational responses about museum artefacts.
    Provides friendly, factual answers based on artefact metadata and type.
    """
    @staticmethod
    def chat(artefact: Artefact, userQuestion: str, history: list = None):
        """
        Generates a conversational response to a user question about a specific artefact.

        The function constructs a prompt using artefact metadata and type, sends it to a language model, 
        and returns a natural-language response. It also maintains a history of the conversation.

        Args:
            artefact (Artefact): The artefact object containing metadata and details.
            userQuestion (str): The question posed by the user about the artefact.
            history (list, optional): List of previous conversation messages (each with "role" and "content").

        Returns:
            A dictionary containing:
                - 'artefact': The ID of the artefact.
                - 'response': The generated chatbot response.
                - 'history': Updated conversation history including the latest exchange.
        """
        if not isinstance(artefact, Artefact):
           return "ERROR: Invalid artefact. Expected an instance of Artefact model."     

        if artefact.metadata.isHF():
            role = ("""You are a cheerful museum guide helping a visitor learn about an event photo.\n
                        This artefact is a photo of people(s) in a event.\n
                        Avoid stereotypes or gender assumptions based on imagery. Be factual and respectful.\n
                    """)
                    
        elif artefact.metadata.isMM():
            role = ("""You are a knowledgeable calligraphy historian assisting a learner.\n
                    This artefact is a photo of handwritten Chinese meeting minutes.\n
                    Do not interpret handwritten text directly unless a transcription is explicitly provided.\n
                    Avoid cultural, political, or gender-based assumptions. Be accurate and respectful.\n
                    """)
        else:
            role = "You are an assistant providing factual information about a museum artefact.\n"

        prompt = "{}\n\nArtefact Details\nName: {}\n".format(role, artefact.name)

        # Check for metadata
        if artefact.metadata is None:
            prompt += "\nThis artefact has no metadata.\n"
        else:
            prompt += "\nThis artefact has the following metadata:\n"
            metadata_repr = artefact.metadata.represent()
            filtered_metadata = InItemChatbot.filterMetadata(metadata_repr)
            if filtered_metadata:
                for key, value in filtered_metadata.items():
                    prompt += "- {}: {}\n".format(key, value or '(Not provided)')
            else:
                prompt += "(No relevant metadata found)\n"

        prompt += """
        Please answer the user's questions by presenting artefact information in a natural, conversational tone. 
        Avoid listing facts robotically; respond as a friendly museum guide would.

        Important guidelines:
        - Base responses strictly on the artefact details provided.
        - Do NOT reference 'metadata' or database terms.
        - If information is missing, explain it politely (e.g., "This information is not documented.").
        - Avoid speculation, assumptions, or interpretation—especially around calligraphy or imagery.
        - If asked unrelated or inappropriate questions, respond with: 
        "I'm here to help with artefact information only. Let's focus on the artefact details."

        Keep your tone clear, engaging, and respectful.
        """
        contextHistory = [
            Interaction(role=Interaction.Role.SYSTEM, content=prompt)
        ]

        # Replay past conversation if exists
        if history:
            for entry in history:
                if "role" in entry and "content" in entry:
                    role = Interaction.Role.USER if entry["role"] == "user" else Interaction.Role.ASSISTANT
                    contextHistory.append(Interaction(role=role, content=entry["content"]))

        # Add current user message
        contextHistory.append(Interaction(role=Interaction.Role.USER, content=userQuestion))

        context = InteractionContext(
            provider=LMProvider.OPENAI,
            variant=LMVariant.GPT_5_NANO,
            history=contextHistory
        )

        result = LLMInterface.engage(context)
        if isinstance(result, str):
            return "ERROR: {}".format(result)
        else:
            botResponse = result.content.strip()

        # Update history
        updatedHistory = (history or []) + [
            {"role": "user", "content": userQuestion},
            {"role": "assistant", "content": botResponse}
        ]

        return {
            "artefact": artefact.id,
            "response": botResponse,
            "history": updatedHistory
        }

    @staticmethod
    def filterMetadata(raw: dict) -> dict:
        """
        Extracts and formats relevant metadata from the raw artefact data.

        Filters out unnecessary fields and structures the metadata to only include 
        relevant values for either calligraphy (MMData) or event photos (HFData).

        Args:
            raw (dict): The raw metadata dictionary from the artefact.

        Returns:
            dict: A dictionary of filtered metadata for use in the LLM prompt.
        """
        if not raw:
            return {}

        if "tradCN" in raw:  # MMData (calligraphy)
            return {
                "Traditional Chinese": raw.get("tradCN"),
                "Simplified Chinese": raw.get("simplifiedCN"),
                "English Translation": raw.get("english"),
                "Summary": raw.get("summary")
            }
        
        elif "faceFiles" in raw:  # HFData (event photos)
            return {
                "Caption": raw.get("caption"),
                "Number of Faces": len(raw.get("faceFiles", []))
            }
        
        return {} 