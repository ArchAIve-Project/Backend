import torch, threading
from database import DI, DIRepresentable, DIError
from fm import File, FileManager
from utils import Ref
from services import Universal, Logger, FileOps
from typing import List, Dict
from faceRecog import FaceEmbedding

class Face(DIRepresentable):
    """Represents a collection of face embeddings associated with a figure, dubbed as a `Face`.
    
    This class sits at the intersection of `DI`, `FileManager` and the facial recognition and detection systems in `faceRecog.py`.
    While having representations for each, individual embedding in the database (`origin` and `added`),
    the model also setups, loads and saves actual `torch` embedding tensors into an `embeddings.pth` (`Face.embeddingsFile`) file in the `people` FileManager store.
    
    Thus, `Face` is a collection of ID-keyed `FaceEmbedding` objects that are populated from 2 data points:
    1. Torch-compatible embeddings file in the `people` store that is backed up by `FileManager`.
    2. The `DatabaseInterface` (DI) NoSQL database that stores non-numerical attributes for each embedding,
    such as `origin` (the image name of the artefact where the embedding was extracted from a face crop) and `added` (the timestamp that the embedding was created).
    
    Attributes:
        embeddings (Dict[str, FaceEmbedding]): A dictionary mapping embedding IDs to FaceEmbedding objects.
        embedsLoaded (bool): Indicates whether the embeddings have been loaded from the file.
        figureID (str): The ID of the figure associated with this Face.
        originRef (Ref): The reference object for database operations, pointing to the figure's Face
    
    Static Attributes:
        embeddingsData (Dict[str, Dict[str, torch.Tensor]] | None): Cached embeddings data loaded from the embeddings file.
        embeddingsFile (File): The file where embeddings are stored, managed by FileManager.
        embedFileLock (threading.RLock): A reentrant lock for thread-safe access to the embeddings file. **Should not be used outside of this class**.
    
    Methods:
        __init__(figure, embeddings, embedsLoaded=False):
            Initializes a Face instance with a figure and embeddings.
        represent():
            Returns a dictionary representation of the Face for serialization.
        save(embeds=True):
            Saves the Face to the database and optionally saves embeddings to the embeddings file.
        addEmbedding(embedding, autoSave=False, embedID=None):
            Adds a FaceEmbedding to the Face.
        loadEmbeds(matchDBEmbeddings=True):
            Loads embeddings from the embeddings file and matches them with the database entries.
        saveEmbeds():
            Saves the current embeddings to the embeddings file.
        offloadEmbeds():
            Clears the embeddings from memory, setting embedsLoaded to False.
        destroy(embeds=True):
            Deletes the Face from the database and optionally removes its embeddings.
        destroyEmbedding(embedID, includingValue=True):
            Removes a specific embedding from the Face.
        reload(withEmbeddings=True, matchDBEmbeddings=True):
            Reloads the Face data from the database, optionally reloading embeddings and matching them with the database entries.
    
    Static Methods:
        ensureEmbeddingsFile():
            Ensures that the embeddings file exists and is ready for use.
        loadEmbeddings():
            Loads the embeddings data from the embeddings file into the cached embeddingsData.
        offloadEmbeddings():
            Clears the cached embeddings data, freeing up memory.
        saveEmbeddings():
            Saves the cached embeddings data to the embeddings file.
            Requires `loadEmbeddings()` to be called first to populate `embeddingsData`.
        extractEmbeddings(figureID: str) -> Dict[str, torch.Tensor]:
            Extracts embeddings for a given figure ID from the cached embeddings data.
            Requires `loadEmbeddings()` to be called first to populate `embeddingsData`.
        extractEmbedding(figureID: str, embeddingID: str) -> torch.Tensor | None:
            Extracts a specific embedding for a given figure ID and embedding ID from the cached embeddings data.
            Requires `loadEmbeddings()` to be called first to populate `embeddingsData`.
        setEmbeddings(figureID: str, embeddings: Dict[str, torch.Tensor]):
            Sets the embeddings for a given figure ID in the cached embeddings data.
            Requires `loadEmbeddings()` to be called first to populate `embeddingsData`.
        setEmbedding(figureID: str, embeddingID: str, embedding: torch.Tensor):
            Sets a specific embedding for a given figure ID and embedding ID in the cached embeddings data.
            Requires `loadEmbeddings()` to be called first to populate `embeddingsData`.
        deleteFigure(figureID: str):
            Deletes the embeddings for a given figure ID from the cached embeddings data.
        deleteEmbedding(figureID: str, embeddingID: str):
            Deletes a specific embedding for a given figure ID and embedding ID from the cached embeddings data.
            Requires `loadEmbeddings()` to be called first to populate `embeddingsData`.
    
    Sample usage:
    ```python
    from fm import FileManager
    from models import DI, Face, FaceEmbedding
    
    DI.setup()
    FileManager.setup() # required as Face uses FileManager for embeddings storage
    
    # Create a new Face with a figure and some embeddings
    figure = "figure123"  # This should be a Figure object or its ID
    embeddings = [
        FaceEmbedding(value=torch.tensor([0.1, 0.2, 0.3]), origin="image1.jpg")
    ]
    
    face = Face(figure=figure, embeddings=embeddings, embedsLoaded=True)
    face.save()  # Save the Face and its embeddings
    
    face = Face.load(figureID="figure123", withEmbeddings=True)  # Load the Face by figure ID with embeddings
    print(face)  # Print the Face representation
    ```
    
    Notes:
        - The `embeddings` attribute is a dictionary mapping embedding IDs to `FaceEmbedding` objects.
        - The `embedsLoaded` attribute indicates whether the embeddings have been loaded from the embeddings file.
        - After making any edits to `Face` or its embeddings, you need to call `save()` to persist changes. Changes are not persisted automatically.
        - The embeddings stored in `Face.embeddingsFile` are in a format compatible with PyTorch, allowing for efficient storage and retrieval of face embeddings. They cannot be opened in direct text editor formats.
    """
    
    embeddingsData: 'Dict[str, Dict[str, torch.Tensor]] | None' = None
    embeddingsFile: 'File | None' = File("embeddings.pth", "people")
    embedFileLock = threading.RLock()
    
    def __init__(self, figure: 'Figure | str', embeddings: 'List[FaceEmbedding] | Dict[str, FaceEmbedding]', embedsLoaded: bool=False):
        """Initialize a Face object.

        Args:
            figure (Figure | str): The figure associated with this face.
            embeddings (List[FaceEmbedding] | Dict[str, FaceEmbedding]): The embeddings for this face. Both lists and dictionaries are supported.
            embedsLoaded (bool, optional): Whether the embeddings have been loaded. Defaults to False.
        """
        
        # Validate embedsLoaded
        if not isinstance(embedsLoaded, bool):
            raise Exception("FACE INIT ERROR: 'embedsLoaded' must be a boolean value.")
        
        # Extract figureID and validate embeddings
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE INIT ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        
        if isinstance(embeddings, list):
            if not all(isinstance(embed, FaceEmbedding) for embed in embeddings):
                raise Exception("FACE INIT ERROR: 'embeddings' must be a list or dictionary of FaceEmbedding objects.")
            
            embeddings = {Universal.generateUniqueID(customLength=6): embedding for embedding in embeddings}
        elif isinstance(embeddings, dict):
            for embed in embeddings.values():
                if not isinstance(embed, FaceEmbedding):
                    raise Exception("FACE INIT ERROR: 'embeddings' must be a list or dictionary of FaceEmbedding objects.")
        else:
            raise Exception("FACE INIT ERROR: 'embeddings' must be a list or dictionary of FaceEmbedding objects.")
        
        # Instantiate Face object
        self.embeddings = embeddings
        self.embedsLoaded = embedsLoaded
        self.figureID = figureID
        self.originRef = Face.ref(figureID)
    
    def represent(self):
        return {
            "embeddings": {embeddingID: embedding.representForDB() for embeddingID, embedding in self.embeddings.items()}
        }
    
    def save(self, embeds=True):
        """Save the Face object and its embeddings.

        Args:
            embeds (bool, optional): Whether to save the embeddings. Defaults to True.

        Returns:
            bool: Almost always `True`.
        """
        
        DI.save(self.represent(), self.originRef)
        
        if not self.embedsLoaded and embeds:
            Logger.log("FACE SAVE WARNING: Embeddings not loaded before saving, saveEmbeds call will be skipped. Figure ID: {}".format(self.figureID))
        elif embeds:
            self.saveEmbeds()
        
        return True
    
    def addEmbedding(self, embedding: FaceEmbedding, autoSave: bool=False, embedID: str=None) -> str:
        """Add an embedding to the face.

        Args:
            embedding (FaceEmbedding): The embedding to add.
            autoSave (bool, optional): Whether to automatically save changes with `self.saveEmbeds()`. Defaults to False.
            embedID (str, optional): The ID to assign to the embedding. Defaults to None.

        Raises:
            Exception: If the embedding is not a FaceEmbedding object.
            Exception: If the embeddings are not loaded.
            Exception: If the embedding ID already exists.

        Returns:
            str: The ID assigned to the embedding.
        """
        
        if not isinstance(embedding, FaceEmbedding):
            raise Exception("FACE ADDEMBEDDING ERROR: 'embedding' must be a FaceEmbedding object.")
        if not self.embedsLoaded:
            raise Exception("FACE ADDEMBEDDING ERROR: Embeddings are not loaded; call loadEmbeds() first.")
        
        if embedID is None:
            embedID = Universal.generateUniqueID(customLength=6, notIn=list(self.embeddings.keys()))
        
        if embedID in self.embeddings:
            raise Exception("FACE ADDEMBEDDING ERROR: Embedding with ID '{}' already exists.".format(embedID))
        
        self.embeddings[embedID] = embedding
        
        if autoSave:
            self.saveEmbeds()
        
        return embedID
    
    def loadEmbeds(self, matchDBEmbeddings: bool=True) -> bool:
        """Load embeddings for the face.

        Args:
            matchDBEmbeddings (bool, optional): Whether to match embeddings with the database, i.e remove embeddings that are not found in the database. Defaults to True.

        Returns:
            bool: Almost always `True`.
        """
        
        if Face.embeddingsData is None:
            Face.loadEmbeddings()
        
        faceEmbeds = Face.extractEmbeddings(self.figureID) or {}
        
        changes = False
        if matchDBEmbeddings:
            # Remove embeddings that are not in the database
            for embeddingID in list(faceEmbeds.keys()):
                if embeddingID not in self.embeddings:
                    del faceEmbeds[embeddingID]
                    changes = True
        
        for embeddingID, embed in self.embeddings.items():
            if isinstance(embed, FaceEmbedding):
                embed.value = faceEmbeds.get(embeddingID, None)
        
        self.embedsLoaded = True
        if changes:
            self.saveEmbeds()
        
        return True
    
    def saveEmbeds(self):
        """Save the embeddings for the face to `Face.embeddingsFile`.

        Raises:
            Exception: If the embeddings are not loaded.

        Returns:
            bool: Almost always `True`.
        """
        
        if not self.embedsLoaded:
            raise Exception("FACE SAVEEMBEDS ERROR: Embeddings are not loaded; call loadEmbeds() first.")
        
        if Face.embeddingsData is None:
            Face.loadEmbeddings()
        
        Face.setEmbeddings(self.figureID, {id: embed.value for id, embed in self.embeddings.items()})
        Face.saveEmbeddings()
        
        return True
    
    def offloadEmbeds(self):
        """Offload the embeddings for the face, i.e clear them from memory. Will set `embedsLoaded` to False.

        Returns:
            bool: Almost always `True`.
        """
        
        for embed in self.embeddings.values():
            embed.value = None
        
        self.embedsLoaded = False
        return True
    
    def destroy(self, embeds: bool=True):
        """Destroy the face and its embeddings data from the database and/or the embeddings file.

        Args:
            embeds (bool, optional): Whether to destroy the embeddings data from `Face.embeddingsFile`. Defaults to True.

        Returns:
            bool: Almost always `True`.
        """
        
        DI.save(None, self.originRef)
        
        if embeds:
            if Face.embeddingsData is None:
                Face.loadEmbeddings()
            
            Face.deleteFigure(self.figureID)
            Face.saveEmbeddings()
        
        return True
    
    def destroyEmbedding(self, embedID: str, includingValue: bool=True):
        """Destroy an embedding from the face from the database and/or the embeddings file.

        Args:
            embedID (str): The ID of the embedding to destroy.
            includingValue (bool, optional): Whether to destroy the embedding from `Face.embeddingsFile`. Defaults to True (recommended).

        Returns:
            bool: Almost always `True`.
        """
        
        if embedID in self.embeddings:
            del self.embeddings[embedID]
        
        if includingValue:
            self.saveEmbeds()
        
        return True
    
    def reload(self, withEmbeddings: bool=True, matchDBEmbeddings: bool=True):
        """Reload the face data from the database and/or the embeddings file.

        Args:
            withEmbeddings (bool, optional): Whether to load the embeddings data as well (impacts `embedsLoaded`). Defaults to True.
            matchDBEmbeddings (bool, optional): Whether to match the embeddings with the database. Defaults to True.

        Raises:
            Exception: If any part of the reload process fails, such as DI load errors or unexpected data formats.

        Returns:
            bool: Almost always `True`.
        """
        
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("FACE RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("FACE RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("FACE RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.figureID, data, withEmbeddings, matchDBEmbeddings).__dict__)
        return True
    
    def __str__(self):
        return "Face(figureID='{}', embeddings={})".format(
            self.figureID,
            {embedID: str(embed) for embedID, embed in self.embeddings.items()} if isinstance(self.embeddings, dict) else None
        )
    
    @staticmethod
    def rawLoad(figureID: str, data: dict, withEmbeddings: bool=True, matchDBEmbeddings: bool=True) -> 'Face':
        """Load a Face object from raw dictionary and embeddings data.

        Args:
            figureID (str): The ID of the figure.
            data (dict): The raw database dictionary to load from.
            withEmbeddings (bool, optional): Whether to load the embeddings data as well from `Face.embeddingsFile` (impacts `embedsLoaded`). Defaults to True.
            matchDBEmbeddings (bool, optional): Whether to match the embeddings with the database. Defaults to True.

        Returns:
            Face: The loaded Face object.
        """
        
        dbEmbeds: Dict[str, Dict[str, str]] = data.get('embeddings', {})
        
        objectEmbeddings: Dict[str, FaceEmbedding] = {}
        for embeddingID, data in dbEmbeds.items():
            if isinstance(data, dict):
                objectEmbeddings[embeddingID] = FaceEmbedding(
                    value=None,
                    origin=data.get('origin', 'Unknown'),
                    added=data.get('added')
                )
        
        f = Face(figureID, objectEmbeddings)
        
        if withEmbeddings:
            f.loadEmbeds(matchDBEmbeddings)
        
        return f

    @staticmethod
    def load(figure: 'Figure | str', withEmbeddings: bool=True, matchDBEmbeddings: bool=True) -> 'Face | None':
        """Load a Face object from the database and/or the embeddings file given a figure object or ID.

        Args:
            figure (Figure | str): The figure object or it's identifier to load the `Face` for.
            withEmbeddings (bool, optional): Whether to load the embeddings data as well from `Face.embeddingsFile` (impacts `embedsLoaded`). Defaults to True.
            matchDBEmbeddings (bool, optional): Whether to match the embeddings with the database. Defaults to True.

        Raises:
            Exception: If any part of the loading process fails.

        Returns: `Face | None`: The loaded Face object or `None` if not found.
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        
        data = DI.load(Face.ref(figureID))
        if data == None:
            if matchDBEmbeddings:
                # If the figureID is not found, check if the embeddings file exists and delete the figure from it
                if Face.embeddingsData is None:
                    Face.loadEmbeddings()
                
                if figureID in Face.embeddingsData:
                    Face.deleteFigure(figureID)
                    Face.saveEmbeddings()
            
            return None
        if isinstance(data, DIError):
            raise Exception("FACE LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("FACE LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return Face.rawLoad(figureID, data, withEmbeddings, matchDBEmbeddings)
    
    @staticmethod
    def ensureEmbeddingsFile():
        """Ensures that an embeddings file exists in the `people` FileManager store and is ready for use.
        
        Sets up a blank file if it does not exist and saves.

        Raises:
            Exception: If a `FileManager` existence check for `Face.embeddingsFile` fails.
            Exception: If `FileManager.prepFile` couldn't prepare the embeddings file for local use.

        Returns:
            bool: Almost always `True`.
        """
        
        fileExists = Face.embeddingsFile.exists()
        
        if isinstance(fileExists, str):
            raise Exception("FACE ENSUREEMBEDDINGSFILE ERROR: Failed to check if embeddings file exists; response: {}".format(fileExists))
        elif not fileExists[0]:
            Face.embeddingsData = {}
            Face.saveEmbeddings()
        elif not fileExists[1] or not FileManager.existsInSystem(Face.embeddingsFile):
            prep = FileManager.prepFile(file=Face.embeddingsFile)
            if not isinstance(prep, File):
                raise Exception("FACE ENSUREEMBEDDINGSFILE ERROR: Failed to prepare embeddings file; response: {}".format(prep))
        
        return True
    
    @staticmethod
    def loadEmbeddings():
        """Loads all embeddings data into memory at `Face.embeddingsData` from the `Face.embeddingsFile`.
        
        For thread-safety, this operation uses a re-entrant lock (`Face.embedFileLock`).

        Raises:
            Exception: If any part of the loading process fails.

        Returns:
            bool: Almost always `True`.
        """
        
        with Face.embedFileLock:
            Face.ensureEmbeddingsFile()
            
            try:
                Face.embeddingsData = torch.load(Face.embeddingsFile.path(), map_location=Universal.device)
            except Exception as e:
                raise Exception("FACE LOADEMBEDDINGS ERROR: Failed to load embeddings data; error: {}".format(e)) from e
        
        return True
    
    @staticmethod
    def offloadEmbeddings(includingFile: bool=False):
        """Offloads embeddings data from memory and optionally from the `people` FileManager store.

        Args:
            includingFile (bool, optional): If True, offloads `Face.embeddingsFile` from the local filesystem as well. Defaults to False.

        Raises:
            Exception: If `includingFile` and `FileManager.offload` fails.

        Returns:
            bool: Almost always `True`.
        """
        
        if Face.embeddingsData != None:
            Face.embeddingsData = None
        
        if includingFile and FileOps.exists(Face.embeddingsFile.path(), "file"):
            res = FileManager.offload(file=Face.embeddingsFile)
            if isinstance(res, str):
                raise Exception("FACE OFFLOADEMBEDDINGS ERROR: Failed to offload embeddings file; response: {}".format(res))
        
        return True
    
    @staticmethod
    def saveEmbeddings():
        """Saves all embeddings data from `Face.embeddingsData` to the `Face.embeddingsFile`. Embeddings data must be loaded first.
        
        For thread-safety, this operation uses a re-entrant lock (`Face.embedFileLock`).

        Raises:
            Exception: If any part of the saving process fails, such as if `Face.embeddingsData` is `None`.

        Returns:
            bool: Almost always `True`.
        """
        
        if Face.embeddingsData is None:
            raise Exception("FACE SAVEEMBEDDINGS ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        with Face.embedFileLock:
            try:
                torch.save(Face.embeddingsData, Face.embeddingsFile.path())
                
                saveResult = FileManager.save(file=Face.embeddingsFile)
                if isinstance(saveResult, str):
                    raise Exception(saveResult)
            except Exception as e:
                raise Exception("FACE SAVEEMBEDDINGS ERROR: Failed to save embeddings data; error: {}".format(e)) from e
        
        return True
    
    @staticmethod
    def extractEmbeddings(figure: 'Figure | str') -> 'Dict[str, torch.Tensor] | None':
        """Extracts all embeddings for a given figure. Embeddings data must be loaded first.

        Args:
            figure (Figure | str): The figure object or its ID.

        Raises:
            Exception: If an invalid `figure` is provided or if `Face.embeddingsData` is not loaded.

        Returns: `Dict[str, torch.Tensor] | None`
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE EXTRACTEMBEDDINGS ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE EXTRACTEMBEDDINGS ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        if not isinstance(Face.embeddingsData, dict):
            return None
        
        return Face.embeddingsData.get(figureID, None)
    
    @staticmethod
    def extractEmbedding(figure: 'Figure | str', embeddingID: str) -> 'torch.Tensor | None':
        """Extracts a specific embedding for a given figure. Embeddings data must be loaded first.

        Args:
            figure (Figure | str): The figure object or its ID.
            embeddingID (str): The ID of the embedding to extract.

        Raises:
            Exception: If an invalid `figure` or `embeddingID` is provided, or if `Face.embeddingsData` is not loaded.

        Returns: `torch.Tensor | None`: The extracted embedding tensor or None if not found.
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE EXTRACTEMBEDDING ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddingID, str):
            raise Exception("FACE EXTRACTEMBEDDING ERROR: 'embeddingID' must be a string representing the embedding ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE EXTRACTEMBEDDING ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        return Face.embeddingsData.get(figureID, {}).get(embeddingID, None)
    
    @staticmethod
    def setEmbeddings(figure: 'Figure | str', embeddings: Dict[str, torch.Tensor]):
        """Sets the embeddings for a given figure. Embeddings data must be loaded first.
        
        Warning: This operation modifies in-memory data only; persist changes by calling `Face.saveEmbeddings()`.

        Args:
            figure (Figure | str): The figure object or its ID.
            embeddings (Dict[str, torch.Tensor]): A dictionary of embeddings to set, keyed by unique IDs.

        Raises:
            Exception: If an invalid `figure` or `embeddings` is provided, or if `Face.embeddingsData` is not loaded.
        
        Returns:
            bool: Almost always `True`.
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE SETEMBEDDINGS ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddings, dict):
            raise Exception("FACE SETEMBEDDINGS ERROR: 'embeddings' must be a dictionary of embeddings.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE SETEMBEDDINGS ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        Face.embeddingsData[figureID] = embeddings
        
        return True
    
    @staticmethod
    def setEmbedding(figure: 'Figure | str', embeddingID: str, embedding: torch.Tensor):
        """Sets a specific embedding for a given figure. Embeddings data must be loaded first.
        
        Warning: This operation modifies in-memory data only; persist changes by calling `Face.saveEmbeddings()`.

        Args:
            figure (Figure | str): The figure object or its ID.
            embeddingID (str): The ID of the embedding to set.
            embedding (torch.Tensor): The embedding tensor to set.

        Raises:
            Exception: If an invalid `figure`, `embeddingID`, or `embedding` is provided, or if `Face.embeddingsData` is not loaded.

        Returns:
            bool: Almost always `True`.
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE SETEMBEDDING ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddingID, str):
            raise Exception("FACE SETEMBEDDING ERROR: 'embeddingID' must be a string representing the embedding ID.")
        if not isinstance(embedding, torch.Tensor):
            raise Exception("FACE SETEMBEDDING ERROR: 'embedding' must be a torch.Tensor object.")
        if Face.embeddingsData is None:
            raise Exception("FACE SETEMBEDDING ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        if figureID not in Face.embeddingsData:
            Face.embeddingsData[figureID] = {}
        
        Face.embeddingsData[figureID][embeddingID] = embedding
        
        return True
    
    @staticmethod
    def deleteFigure(figure: 'Figure | str'):
        """Deletes a figure from embeddings data. Embeddings data must be loaded first.
        
        Warning: This operation modifies in-memory data only; persist changes by calling `Face.saveEmbeddings()`.

        Args:
            figure (Figure | str): The figure object or its ID.

        Raises:
            Exception: If an invalid `figure` is provided, or if `Face.embeddingsData` is not loaded.

        Returns:
            bool: Almost always `True`.
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE DELETEFIGURE ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE DELETEFIGURE ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        if figureID in Face.embeddingsData:
            del Face.embeddingsData[figureID]
        
        return True
    
    @staticmethod
    def deleteEmbedding(figure: 'Figure | str', embeddingID: str):
        """Deletes an embedding from a figure's embeddings data. Embeddings data must be loaded first.
        
        Warning: This operation modifies in-memory data only; persist changes by calling `Face.saveEmbeddings()`.

        Args:
            figure (Figure | str): The figure object or its ID.
            embeddingID (str): The ID of the embedding to delete.

        Raises:
            Exception: If an invalid `figure` or `embeddingID` is provided, or if `Face.embeddingsData` is not loaded.

        Returns:
            bool: Almost always `True`.
        """
        
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE DELETEEMBEDDING ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddingID, str):
            raise Exception("FACE DELETEEMBEDDING ERROR: 'embeddingID' must be a string representing the embedding ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE DELETEEMBEDDING ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        if figureID in Face.embeddingsData and embeddingID in Face.embeddingsData[figureID]:
            del Face.embeddingsData[figureID][embeddingID]
        
        return True
    
    @staticmethod
    def ref(figureID: str) -> Ref:
        return Ref("figures", figureID, "face")

class Figure(DIRepresentable):
    """A data model that represents a Figure, i.e a person who has appeared in artefact images.
    
    Figures are meant to be unique database objects for each person that appears in any artefact across the catalogue.
    Each figure has a label, a headshot image, and a Face object that contains embeddings and numerical information about the figure's face.
    
    The model is primarily part of the data import workflow, where the face crop matching algorithm in `HFProcessor` creates new `Figure` objects. This data model
    also supports Face Recognition and Auto Profile features.
    
    The `Face` object is closely linked to `Figure`, and is stored as the `face` attribute. It primarily contains embeddings and other numerical data about the figure's face
    that is useful for face detection and comparison.
    
    The `headshot` attribute is meant to be the image name in the `people` FileManager store; the image should be a "headshot"/"face crop" of the figure.
    
    The `label` might default to the figure's ID, but it is recommended to update it to the figure's actual name, which can be user-provided.
    
    Attributes:
        id (str): Unique identifier for the figure.
        label (str): Name or label of the figure.
        headshot (str): Name of the headshot image file in the `people` FileManager store
        face (Face): Face object containing embeddings and numerical data about the figure's face.
        profile (None): Placeholder for a profile object, if applicable.
        originRef (Ref): Reference object for database operations, pointing to the figure's data in DI.
    
    Methods:
        __init__(label: str, headshot: str, face: Face, profile: None=None, id: str=None):
            Initializes a Figure instance with the provided label, headshot, face, profile, and ID.
        represent() -> dict:
            Returns a dictionary representation of the Figure instance for serialization.
        save(embeds: bool=True):
            Saves the Figure instance to the database, optionally saving embeddings if they are loaded.
        destroy(embeds: bool=True):
            Deletes the Figure instance from the database, optionally deleting associated embeddings.
        reload(withEmbeddings: bool=True, matchDBEmbeddings: bool=True):
            Reloads the Figure instance from the database, updating its attributes and optionally loading embeddings.
        rawLoad(figureID: str, data: dict, withEmbeddings: bool=False, matchDBEmbeddings: bool=True) -> 'Figure':
            Loads a Figure instance from raw data, including face embeddings if specified.
        load(id: str=None, label: str=None, withEmbeddings: bool=False, matchDBEmbeddings: bool=True) -> 'Figure | List[Figure] | None':
            Loads a Figure instance by ID or label, optionally loading embeddings.
        ref(id: str) -> Ref:
            Returns a reference object for the Figure instance, used for database operations.
    
    Sample usage:
    ```python
    import torch
    from fm import FileManager
    from models import DI, Figure, Face
    
    DI.setup()
    FileManager.setup() # required as FileManager is used by the Face model
    
    # Create a new figure with a headshot and face embeddings
    figure = Figure(label="John Doe", headshot="john_doe.jpg", face=None) # face=None will initialise an empty Face object
    figure.face = Face(figure=figure, embeddings=[FaceEmbedding(value=torch.randn(1, 512), origin="sourceArtefact.jpg")], embedsLoaded=True) # Create a Face object with sample embeddings
    figure.save(embeds=True)  # Save the figure and its face embeddings
    
    # Load an existing figure by ID
    loaded_figure = Figure.load(figure.id, withEmbeddings=True)
    print(loaded_figure.represent())
    ```
    
    Notes:
        - The `label` should be updated to the figure's actual name after creation.
        - The `headshot` should be a valid image file name in the `people` FileManager store.
        - The `face` attribute is crucial for face recognition and should be properly initialized with embeddings.
        - The `profile` attribute is currently a placeholder and can be extended to include additional profile information if needed.
    """
    
    def __init__(self, label: str, headshot: str, face: Face, profile: None=None, id: str=None):
        """Initializes a Figure instance.

        Args:
            label (str): The name or label of the figure.
            headshot (str): The file name of the figure's headshot image.
            face (Face): The Face object containing embeddings and data about the figure's face. Defaults to an empty Face object if `None` is provided.
            profile (None, optional): Placeholder for a profile object, if applicable. Defaults to None.
            id (str, optional): Unique identifier for the figure. Defaults to a new unique ID if not provided.
        
        Raises:
            Exception: If the `face` parameter is not a Face object or `None`.
        """
        
        if id is None:
            id = Universal.generateUniqueID()
        if face is None:
            face = Face(id, [], embedsLoaded=True)
        elif not isinstance(face, Face):
            raise Exception("FIGURE INIT ERROR: 'face' must be a Face object.")
        
        self.id = id
        self.label = label
        self.headshot = headshot
        self.face = face
        self.profile = profile
        self.originRef = Figure.ref(id)
    
    def represent(self) -> dict:
        return {
            "label": self.label,
            "headshot": self.headshot,
            "face": self.face.represent() if isinstance(self.face, Face) else None,
            "profile": self.profile
        }
    
    def save(self, embeds: bool=True):
        """Saves the figure and its associated data. Will save to database and embeddings file (through `Face.saveEmbeds`) by default.

        Args:
            embeds (bool, optional): Whether to save embeddings. Defaults to True.

        Returns:
            bool: Almost always `True`.
        """
        
        DI.save(self.represent(), self.originRef)
        
        if not self.face.embedsLoaded and embeds:
            Logger.log("FIGURE SAVE WARNING: Embeddings not loaded before saving, saveEmbeds call will be skipped. Figure ID: {}".format(self.id))
        elif embeds:
            self.face.saveEmbeds()
        
        return True
    
    def destroy(self, embeds: bool=True):
        """Destroys the figure and its associated data. Deletes embeddings data and database entry by default.

        Args:
            embeds (bool, optional): Whether to delete embeddings. Defaults to True.

        Returns:
            bool: Almost always `True`.
        """
        
        DI.save(None, self.originRef)
        
        if embeds:
            if Face.embeddingsData is None:
                Face.loadEmbeddings()
            
            Face.deleteFigure(self.id)
            Face.saveEmbeddings()
        
        return True
    
    def reload(self, withEmbeddings: bool=True, matchDBEmbeddings: bool=True):
        """Reloads the figure and its associated data.

        Args:
            withEmbeddings (bool, optional): Whether to reload embeddings. Defaults to True.
            matchDBEmbeddings (bool, optional): Whether to match database embeddings. Defaults to True.

        Returns:
            bool: Almost always `True`.
        """
        
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("FIGURE RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("FIGURE RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("FIGURE RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.id, data, withEmbeddings, matchDBEmbeddings).__dict__)
        return True
    
    @staticmethod
    def rawLoad(figureID: str, data: dict, withEmbeddings: bool=False, matchDBEmbeddings: bool=True) -> 'Figure':
        """Loads a figure from the provided data dictionary.

        Args:
            figureID (str): The ID of the figure to load.
            data (dict): The data dictionary containing figure information.
            withEmbeddings (bool, optional): Whether to load embeddings. Defaults to False.
            matchDBEmbeddings (bool, optional): Whether to match database embeddings. Defaults to True.

        Returns:
            Figure: The loaded figure.
        """
        
        reqParams = ['label', 'headshot', 'face', 'profile']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'face':
                    data[reqParam] = {}
                else:
                    data[reqParam] = None
        
        face = None
        if isinstance(data['face'], dict):
            face = Face.rawLoad(figureID, data['face'], withEmbeddings, matchDBEmbeddings)
        
        return Figure(
            label=data['label'],
            headshot=data['headshot'],
            face=face,
            profile=data['profile'],
            id=figureID
        )
    
    @staticmethod
    def load(id: str=None, label: str=None, withEmbeddings: bool=False, matchDBEmbeddings: bool=True) -> 'Figure | List[Figure] | None':
        """Loads a figure or figures based on the provided parameters.

        Args:
            id (str, optional): The ID of the figure to load. Defaults to None.
            label (str, optional): The label of the figure to load. Defaults to None.
            withEmbeddings (bool, optional): Whether to load embeddings. Defaults to False.
            matchDBEmbeddings (bool, optional): Whether to match database embeddings. Defaults to True.

        Raises:
            Exception: If there is an error loading the figure data from the database or if the response format is unexpected.

        Returns: `Figure | List[Figure] | None`: The loaded figure or `None` if unique identifiers are provided, otherwise a list of figures (that may be empty).
        """
        
        if id != None:
            data = DI.load(Figure.ref(id))
            if data is None:
                if label is not None:
                    return Figure.load(label=label, withEmbeddings=withEmbeddings, matchDBEmbeddings=matchDBEmbeddings)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("FIGURE LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("FIGURE LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            return Figure.rawLoad(id, data, withEmbeddings, matchDBEmbeddings)
        else:
            data = DI.load(Ref("figures"))
            if data is None:
                return [] if label is None else None
            if isinstance(data, DIError):
                raise Exception("FIGURE LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("FIGURE LOAD ERROR: Failed to load dictionary figures data; response: {}".format(data))

            figures: Dict[str, Figure] = {}
            for id in data:
                if isinstance(data[id], dict):
                    figures[id] = Figure.rawLoad(id, data[id])

            if label is None:
                if withEmbeddings:
                    for figure in figures.values():
                        figure.face.loadEmbeds(matchDBEmbeddings)
                
                return list(figures.values())
            
            for figure in figures.values():
                if label is not None and figure.label == label:
                    if withEmbeddings:
                        figure.face.loadEmbeds(matchDBEmbeddings)
                    
                    return figure
            
            return None
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("figures", id)