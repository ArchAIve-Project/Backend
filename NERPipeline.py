import nltk
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.tokenize import wordpunct_tokenize
from addons import ModelStore, ASTracer, ASReport, ArchSmith

class NERPipeline:
    """
    Named Entity Recognition (NER) pipeline using NLTK.
    Identifies entities like PERSON, ORGANIZATION, LOCATION from English text using
    traditional NLP techniques.
    """
    loaded = False

    @staticmethod
    def load_model(model_path=None):
        """
        Load all required NLTK resources for tokenization, POS tagging, and named entity chunking.
        This method is called once during model setup.
        """
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("maxent_ne_chunker_tab", quiet=True)
        nltk.download("words", quiet=True)

        NERPipeline.loaded = True
        return True

    @staticmethod
    def extract_entities(tree):
        """
        Extract named entities from an nltk.Tree output of ne_chunk().
        Returns a list of tuples (entity_text, entity_type).
        """
        entities = []
        for subtree in tree:
            if isinstance(subtree, Tree):
                entity_label = subtree.label()
                entity_text = " ".join(token for token, _ in subtree.leaves())
                entities.append((entity_text, entity_label))
        return entities

    @staticmethod
    def predict(sentence: str, tracer: ASTracer):
        """
        Predict named entities in the input sentence.
        Steps:
        1. Tokenize the sentence into words.
        2. Perform POS tagging.
        3. Apply named entity chunking.
        4. Extract and return named entities.
        Also logs result to the tracer.
        """
        if not NERPipeline.loaded:
            return "ERROR: NER pipeline has not been loaded. Call load_model() first."

        words = wordpunct_tokenize(sentence)
        tagged = pos_tag(words)
        chunked = ne_chunk(tagged)
        entities = NERPipeline.extract_entities(chunked)

        tracer.addReport(ASReport(
            source="NERPipeline (NLTK)",
            message=f"Extracted {len(entities)} entities",
            extraData={"sentence": sentence}
        ))

        return entities


# ========== Entry Point for Standalone Testing ==========
if __name__ == "__main__":
    # Setup and register the NER pipeline into ModelStore
    ModelStore.setup(ner=NERPipeline.load_model)

    # Create a new tracer to capture logs
    tracer = ArchSmith.newTracer("NER testing:")

    # Example sentence for testing NER
    sentence = """
    On September 14, 2023, at 10:30 AM, President Emmanuel Macron met with Elon Musk and Dr. Angela Zhang 
    at the United Nations headquarters in New York City to discuss AI regulations. The World Health Organization (WHO), 
    in collaboration with the U.S. Department of Health and Human Services, announced a $5 billion initiative to combat 
    future pandemics. Meanwhile, Apple Inc. is planning to acquire Neuralink for approximately $2.5 billion, with a 
    10% stock option to be distributed to early investors. The meeting was streamed live from the Tesla Gigafactory 
    in Nevada and translated into 12 languages. Later that day, the group toured the Statue of Liberty, Times Square, 
    and the Hudson River Bridge. A press conference was scheduled at the Eiffel Tower in Paris the following week. 
    In a separate development, the European Union granted 25% tax relief to green tech companies operating in 
    Germany, Denmark, and Sweden.
    """

    # Run the NER pipeline and print results
    entities = NERPipeline.predict(sentence=sentence, tracer=tracer)
    print(entities)

    # Finalize and persist tracer logs
    tracer.end()
    ArchSmith.persist()
