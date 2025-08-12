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
    
    @staticmethod
    def fill(art: Artefact, categories: List[Category]):
        categories = categories or []
        if not CategorisationPrompt.text:
            return None
        
        filledText = CategorisationPrompt.text
        
        existingCategories = ['- {}: {}'.format(c.name, c.description) for c in categories]
        existingCategories = "\n".join(existingCategories) or "No categories yet"
        
        filledText = filledText.format(
            existingCategories=existingCategories,
            artName=art.name,
            artCaption=art.metadata.raw.caption,
            artAddInfo=art.metadata.raw.addInfo
        )
        
        return filledText

class AutoCategoriser:
    @staticmethod
    def getLLMCategorisation(art: Artefact, categories: List[Category]) -> None:
        cont = InteractionContext(
            provider=LMProvider.QWEN,
            variant=LMVariant.QWEN_VL_PLUS
        )
        
        if not CategorisationPrompt.text:
            CategorisationPrompt.load()
        prompt = CategorisationPrompt.fill(art, categories)
        if not prompt:
            print("Error: Failed to generate prompt")
            return
        
        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content=prompt
            )
        )

        out = LLMInterface.engage(cont)

        return out.content
    
    @staticmethod
    def feed(art: Artefact):
        categories = Category.load()
        
        return AutoCategoriser.getLLMCategorisation(art, categories)