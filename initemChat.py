from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant
from models import Artefact
import os, sys

def chatAboutArtefact(artefactID: str, userQuestion: str):
    artefact = Artefact.load(id=artefactID)
    if artefact is None:
        return "Artefact with ID {} not found.".format(artefactID)

    prompt = """
    You are an assistant for artefact information.
    - Name: {}
    - Created: {}
    - Public: {}
    """.format(artefact.name, artefact.created, artefact.public)

    # Check for metadata
    if artefact.metadata is None:
        prompt += "\nThis artefact has no metadata.\n"
    else:

        if artefact.metadata.isHF():
            prompt = "You are a cheerful museum guide helping a visitor learn about an event photo.\n"
        elif artefact.metadata.isMM():
            prompt = "You are a knowledgeable calligraphy historian assisting a learner.\n"

        prompt += "\nThis artefact has the following metadata:\n"
        metadata_repr = artefact.metadata.represent()
        filtered_metadata = filterMetadata(metadata_repr)
        if filtered_metadata:
            for key, value in filtered_metadata.items():
                prompt += "  - {}: {}\n".format(key, value)
                if not value:
                    prompt += "  - {}: (Not provided)\n".format(key)
        else:
            prompt += "  (No relevant metadata found)\n"

    prompt += "\nAnswer the user's question based on the artefact details above."

    # InteractionContext setup
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
    return "LLM Reply: {}".format(result.content)

def filterMetadata(raw: dict) -> dict:
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


if __name__ == "__main__":
    # Setup LLMInterface
    res = LLMInterface.initDefaultClients()
    if res != True:
        print("MAIN BOOT: Failed to initialise default clients for LLMInterface; response:", res)
        sys.exit(1)
    print("LLMINTERFACE: Default clients initialised successfully.")

    reply = chatAboutArtefact(
        "acf2c779e50c44e5ba97d2449f3956d6",
        "What can you tell me about this artefact?"
    )
    print(reply)

    while True:
        try:
            exec(input(">>> "))
        except Exception as e:
            print(f"Error: {e}")