import re
from typing import List, Dict
from services import Universal, Logger
from addons import ArchSmith, ASReport, ASTracer
from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant
from schemas import Batch, BatchArtefact, Artefact, Category, CategoryArtefact

LLMInterface.initDefaultClients()

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
    def parse_llm_output(text):
        # Variant 1: existing category
        variant_1_pattern = re.compile(
            r"^CATEGORY:\s*(?P<category>.+?)\s*"
            r"REASON:\s*(?P<reason>.+)$",
            re.DOTALL | re.IGNORECASE
        )

        # Variant 2: existing category with optional revised fields
        variant_2_pattern = re.compile(
            r"^CATEGORY:\s*(?P<category>.+?)\s*"
            r"(?:CATEGORY_NAME_REVISED:\s*(?P<name_revised>.+?)\s*)?"
            r"(?:CATEGORY_DESC_REVISED:\s*(?P<desc_revised>.+?)\s*)?"
            r"REASON:\s*(?P<reason>.+)$",
            re.DOTALL | re.IGNORECASE
        )

        # Variant 3: new category
        variant_3_pattern = re.compile(
            r"^NEW_CATEGORY:\s*(?P<new_category>.+?)\s*"
            r"CATEGORY_DESC:\s*(?P<category_desc>.+?)\s*"
            r"REASON:\s*(?P<reason>.+)$",
            re.DOTALL | re.IGNORECASE
        )
        
        text = text.strip()

        match = variant_2_pattern.match(text)
        if match and (match.group("name_revised") or match.group("desc_revised")):
            return {
                "variant": 2,
                "category": match.group("category").strip(),
                "category_name_revised": match.group("name_revised").strip() if match.group("name_revised") else None,
                "category_desc_revised": match.group("desc_revised").strip() if match.group("desc_revised") else None,
                "reason": match.group("reason").strip()
            }

        match = variant_1_pattern.match(text)
        if match:
            return {
                "variant": 1,
                "category": match.group("category").strip(),
                "reason": match.group("reason").strip()
            }

        match = variant_3_pattern.match(text)
        if match:
            return {
                "variant": 3,
                "new_category": match.group("new_category").strip(),
                "category_desc": match.group("category_desc").strip(),
                "reason": match.group("reason").strip()
            }

        raise ValueError("Output does not match any known variant format.")
    
    @staticmethod
    def getLLMCategorisation(art: Artefact, categories: List[Category], tracer: ASTracer, tries: int=3) -> None:
        cont = InteractionContext(
            provider=LMProvider.OPENAI,
            variant=LMVariant.GPT_5_NANO
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
        
        print("PRE TRIALS")
        print(cont)
        
        out = None
        parsedOutput = None
        while tries > 0:
            out = LLMInterface.engage(cont)
            
            print("AFTER ENGAGE")
            print(cont)
            
            if isinstance(out, str):
                Logger.log("AUTOCATEGORISER GETLLMCATEGORISATION ERROR: Failed to categorise artefact '{}'; response: {}".format(art.id, out))
                tracer.addReport(
                    ASReport(
                        source="AUTOCATEGORISER GETLLMCATEGORISATION ERROR",
                        message="Couldn't engage LLM to categorise artefact with ID '{}'; response: {}".format(art.id, out),
                        extraData={
                            "response": out,
                            "artefactID": art.id
                        }
                    )
                )
                return None
            
            
            try:
                parsedOutput = AutoCategoriser.parse_llm_output(out.content)
                break
            except Exception as e:
                tracer.addReport(
                    ASReport(
                        source="AUTOCATEGORISER GETLLMCATEGORISATION",
                        message="Failed to parse LLM output for artefact '{}'; error: {}".format(art.id, e),
                        extraData={"retries": tries}
                    )
                )
                tries -= 1
                cont.addInteraction(
                    Interaction(
                        role=Interaction.Role.USER,
                        content="Your response was not in the expected format. Try again and stick to the given output format instructions."
                    )
                )
                
                continue
        
        if not parsedOutput:
            tracer.addReport(
                ASReport(
                    source="AUTOCATEGORISER GETLLMCATEGORISATION ERROR",
                    message="Valid categorisation output for artefact '{}' couldn't be obtained from LLM within reasonable tries.".format(art.id)
                )
            )
            return None
        
        tracer.addReport(
            ASReport(
                source="AUTOCATEGORISER GETLLMCATEGORISATION",
                message="Successfully categorised artefact '{}' with LLM".format(art.id),
                extraData={
                    "artefactID": art.id,
                    "output": parsedOutput
                }
            )
        )
        
        print("POST TRIALS")
        print(cont)
        
        return parsedOutput
    
    @staticmethod
    def feed(art: Artefact, categories: List[Category], tracer: ASTracer):
        return AutoCategoriser.getLLMCategorisation(art, categories, tracer)