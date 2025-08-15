from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant
from schemas import Artefact

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
        role = ""

        if not isinstance(artefact, Artefact):
           return "ERROR: Invalid artefact. Expected an instance of Artefact model."     

        if artefact.metadata.isHF():
            role = "You are Archivus — a friendly companion who provides interesting and factual information about this artefact."
            role += "This artefact is a photo of people at an event."
            role += "Avoid stereotypes or gender assumptions based on imagery."
            role += "Be curious, welcoming, and engaging, but stay factual."
            role += "If details aren’t available, say so kindly and share what is known."

        elif artefact.metadata.isMM():
            role = "You are Archivus — a thoughtful and friendly assistant knowledgeable about calligraphy."
            role += "This artefact is a photo of handwritten Chinese meeting minutes."
            role += "Share interesting context about calligraphy and its history when relevant,"
            role += "but don’t interpret or translate handwriting unless given a transcription."
            role += "Be approachable, respectful, and focus on what is documented."
        else:
            role += "You are Archivus — a friendly, knowledgeable museum companion."
            role += "You help visitors learn about this artefact in a conversational way,"
            role += "weaving facts into an engaging story without making things up."

        prompt = "{}\n\nArtefact Details\nName: {}\n".format(role, artefact.name)

        # Check for metadata
        if artefact.metadata is None:
            prompt += "\nThis artefact has no metadata.\n"
        else:
            prompt += "\nThis artefact has the following metadata:\n"
            metadataRepr = artefact.metadata.represent()
            filteredMetadata = InItemChatbot.filterMetadata(metadataRepr)
            if filteredMetadata:
                for value in filteredMetadata.items():
                    prompt += "- {}\n".format(value or '(Not provided)')
            else:
                prompt += "(No relevant metadata found)\n"

        prompt += "Please answer the user's questions in a natural, friendly, and engaging tone."
        prompt += "Weave the information into a conversational style—avoid listing facts robotically or using technical terms like 'metadata'."
        prompt += "When answering:"
        prompt += "- Speak as if talking to a curious visitor."
        prompt += "- If something is missing, acknowledge it politely, e.g., 'Unfortunately, we don’t have that detail right now, but here’s what we do know.'"
        prompt += "- Stay focused on this artefact and steer back if the question is unrelated."
        prompt += "- If the user’s question repeats something already explained, briefly acknowledge it and then share the next interesting detail."
        prompt += "- Where appropriate, ask a follow-up question to continue the discussion naturally."
        prompt += "- Avoid speculation, assumptions, or interpretation, especially around calligraphy or imagery."

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
        
        elif "figureIDs" in raw:  # HFData (event photos)
            return {
                "Caption": raw.get("caption"),
                "Additional Information": raw.get("addInfo")
            }
        
        return {}