from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant
from models import Artefact
import sys

class ArtefactChatbot:
    """
    A chatbot for generating conversational responses about museum artefacts.
    Provides friendly, factual answers based on artefact metadata and type.
    
    Example usage:
    bot = ArtefactChatbot()
    response = bot.chat("A123", "What can we learn from this artefact?")
    print(response)
    """
    def chat(self, artefactID: str, userQuestion: str):
        """
        Generates a conversational response to a user question about a specific artefact.

        This function loads an artefact by its ID, prepares a tailored prompt using its metadata and type, 
        and queries a language model to answer the question in a museum guide style. 
        Responses avoid speculation, stereotypes, or mention of metadata fields explicitly.

        Args:
            artefactID (str): The unique identifier of the artefact in the database.
            userQuestion (str): The user's question about the artefact.

        Returns:
            str: A response based on artefact data, or an error message if the artefact is not found or the model fails.
        """
        artefact = Artefact.load(id=artefactID)
        if artefact is None:
            return "Artefact with ID {} not found.".format(artefactID)

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

        prompt = "{}\n\nArtefact Details\n Name: {}\n".format(role, artefact.name)

        # Check for metadata
        if artefact.metadata is None:
            prompt += "\nThis artefact has no metadata.\n"
        else:
            prompt += "\nThis artefact has the following metadata:\n"
            metadata_repr = artefact.metadata.represent()
            filtered_metadata = self.filterMetadata(metadata_repr)
            if filtered_metadata:
                for key, value in filtered_metadata.items():
                    if value:
                        prompt += "- {}: {}\n".format(key, value)
                    else:
                        prompt += "- {}: (Not provided)\n".format(key)
            else:
                prompt += "(No relevant metadata found)\n"

        prompt += """
        Please answer the user's questions by presenting artefact information in a natural, conversational tone. 
        Avoid listing facts robotically—respond as a friendly museum guide would.

        Important guidelines:
        - Base responses strictly on the artefact details provided.
        - Do NOT reference 'metadata' or database terms.
        - If information is missing, explain it politely (e.g., "This information is not documented.").
        - Avoid speculation, assumptions, or interpretation—especially around calligraphy or imagery.
        - If asked unrelated or inappropriate questions, respond with: 
        "I'm here to help with artefact information only. Let's focus on the artefact details."

        Keep your tone clear, engaging, and respectful.
        """

        context = InteractionContext(
            provider=LMProvider.OPENAI,
            variant=LMVariant.GPT_4O_MINI,
            history=[
                Interaction(role=Interaction.Role.SYSTEM, content=prompt),
                Interaction(role=Interaction.Role.USER, content=userQuestion)
            ]
        )

        result = LLMInterface.engage(context)
        if isinstance(result, str):
            return "Error: {}".format(result)
        return "{}".format(result.content)

    def filterMetadata(self, raw: dict) -> dict:
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