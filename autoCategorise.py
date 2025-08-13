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
        if not CategorisationPrompt.text:
            return None
        
        filledText = CategorisationPrompt.text
        
        existingCategories = ['- {}: {}'.format(c.name, c.description) for c in categories]
        existingCategories = "\n".join(existingCategories) or "No categories yet"
        artCaption = "No caption available" if not (art.metadata and art.metadata.raw) else (art.metadata.raw.caption or "No caption available")
        artAddInfo = "No additional information available" if not (art.metadata and art.metadata.raw) else (art.metadata.raw.addInfo or "No additional information available")
        
        filledText = filledText.format(
            existingCategories=existingCategories,
            artName=art.name or "No name available",
            artCaption=artCaption,
            artAddInfo=artAddInfo
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
        
        out = None
        parsedOutput = None
        while tries > 0:
            out = LLMInterface.engage(cont)
            
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
                        source="AUTOCATEGORISER GETLLMCATEGORISATION WARNING",
                        message="Failed to parse LLM output for artefact '{}'; error: {}".format(art.id, e),
                        extraData={"retries": tries, "output": out.content}
                    )
                )
                tries -= 1
                cont.addInteraction(
                    Interaction(
                        role=Interaction.Role.USER,
                        content="Your response was not in the expected format. Remember to NOT provide the case number or description.\n\nTry again and stick to the given output format instructions for any one of the three cases."
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
        
        return parsedOutput
    
    @staticmethod
    def feed(art: Artefact, categories: List[Category], tracer: ASTracer):
        if not isinstance(categories, list) or not all(isinstance(c, Category) for c in categories):
            return None
        
        result = AutoCategoriser.getLLMCategorisation(art, categories, tracer)
        if not isinstance(result, dict):
            tracer.addReport(
                ASReport(
                    source="AUTOCATEGORISER FEED ERROR",
                    message="Valid result couldn't be obtained in LLM categorisation attempt for artefact {}.".format(art.id),
                    extraData={"result": result}
                )
            )
            return None
        
        if result['variant'] in [1, 2]:
            targetCategory = [x for x in categories if x.name == result['category']]
            if not targetCategory:
                tracer.addReport(
                    ASReport(
                        "AUTOCATEGORISER FEED ERROR",
                        "No such category '{}' found based on LLM output.".format(result['category'])
                    )
                )
                return None
            
            targetCategory = targetCategory[0]
            reason = result.get('reason', None)
            if not reason:
                tracer.addReport(
                    ASReport(
                        "AUTOCATEGORISER FEED ERROR",
                        "No reason provided in LLM output for artefact '{}'.".format(art.id)
                    )
                )
                return None
            
            if art.id not in targetCategory.members:
                targetCategory.add(art, reason)
            else:
                targetCategory.members[art.id].reason = reason
            
            changes = {}
            revisedName = result.get('name_revised', None)
            revisedDesc = result.get('desc_revised', None)
            if revisedName:
                if revisedName in [x.name for x in categories]:
                    tracer.addReport(
                        ASReport(
                            "AUTOCATEGORISER FEED WARNING",
                            "Revised category name '{}' already exists, will not update name.".format(revisedName),
                            extraData={"reason": reason}
                        )
                    )
                else:
                    changes['categoryName'] = "{} -> {}".format(targetCategory.name, revisedName)
                    targetCategory.name = revisedName
            
            if revisedDesc:
                changes['categoryDesc'] = "{} -> {}".format(targetCategory.description, revisedDesc)
                targetCategory.description = revisedDesc
            
            try:
                targetCategory.save()
            except Exception as e:
                tracer.addReport(
                    ASReport(
                        "AUTOCATEGORISER FEED ERROR",
                        "Failed to save category '{}' after updating artefact '{}'; error: {}".format(targetCategory.name, art.id, e)
                    )
                )
                return None
            
            tracer.addReport(
                ASReport(
                    "AUTOCATEGORISER FEED",
                    "Successfully categorised artefact '{}' to category with ID '{}' and name '{}'.".format(art.id, targetCategory.id, targetCategory.name),
                    extraData={"artefactID": art.id, "categoryID": targetCategory.id, "changes": changes, "reason": reason}
                )
            )
            return targetCategory.id
        else:
            newCatName = result.get('new_category', None)
            newCatDesc = result.get('category_desc', None)
            reason = result.get('reason', None)
            
            if not (newCatName and newCatDesc and reason):
                tracer.addReport(
                    ASReport(
                        "AUTOCATEGORISER FEED ERROR",
                        "Malformed data for variant 3 processing in categorisation for artefact '{}'.".format(art.id)
                    )
                )
                return None
            if newCatName in [x.name for x in categories]:
                tracer.addReport(
                    ASReport(
                        "AUTOCATEGORISER FEED ERROR",
                        "Category name '{}' already exists, will not create new category for variant 3 processing.".format(newCatName),
                        extraData={"artefactID": art.id, "reason": reason}
                    )
                )
                return None

            newCategory = Category(
                name=newCatName,
                description=newCatDesc
            )
            
            newCategory.add(art, reason)
            
            try:
                newCategory.save()
            except Exception as e:
                tracer.addReport(
                    ASReport(
                        "AUTOCATEGORISER FEED ERROR",
                        "Failed to save new category '{}' after adding artefact '{}'; error: {}".format(newCategory.name, art.id, e)
                    )
                )
                return None
            
            categories.append(newCategory)
            
            tracer.addReport(
                ASReport(
                    "AUTOCATEGORISER FEED",
                    "Successfully created new category '{}' with ID '{}' and assigned artefact '{}' as member.".format(newCategory.name, newCategory.id, art.id),
                    extraData={"artefactID": art.id, "categoryID": newCategory.id, "reason": reason}
                )
            )
            return newCategory.id