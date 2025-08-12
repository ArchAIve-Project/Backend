from typing import List, Dict
from services import Universal, Logger
from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant
from schemas import Batch, BatchArtefact, Artefact, Category, CategoryArtefact

class CategorisationPrompt:
    text = None
    
    @staticmethod
    def load():
        with open("templates/llmCategorisationPrompt.txt", "r") as f:
            CategorisationPrompt.text = f.read()

class AutoCategoriser:
    @staticmethod
    def getLLMCategorisation(art: Artefact, categories: List[Category]) -> None:
        cont = InteractionContext(
            provider=LMProvider.QWEN,
            variant=LMVariant.QWEN_VL_PLUS
        )

        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content="TBC"
            )
        )

        out = LLMInterface.engage(cont)

        print(out.content)
    
    @staticmethod
    def feed(art: Artefact):
        pass