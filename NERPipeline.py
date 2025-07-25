import nltk
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.tokenize import wordpunct_tokenize
from addons import ASTracer, ASReport

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
            source="NERPIPELINE PREDICT",
            message="Extracted {} entities".format(len(entities)),
            extraData={"sentence": sentence}
        ))

        return entities