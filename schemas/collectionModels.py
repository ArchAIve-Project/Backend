from enum import Enum
from database import DI, DIRepresentable, DIError
from utils import Ref
from services import Universal, Logger
from typing import List, Dict, Any, Literal
from .userModels import User
from .artefactModels import Artefact

class Book(DIRepresentable):
    def __init__(self, title: str, subtitle: str, mmIDs: List[str], id: str=None):
        """
        Represents a curated collection of MMData artefacts as a Book.

        Args:
            title (str): The title of the book.
            subtitle (str): Subtitle or additional description.
            mmIDs (List[str]): A list of artefact IDs that are part of the book.
            id (str, optional): Unique ID for the book. Auto-generated if not provided.
        """
        if id is None:
            id = Universal.generateUniqueID()
        
        self.id = id
        self.title = title
        self.subtitle = subtitle
        self.mmIDs = mmIDs
        self.originRef = Book.ref(self.id)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "mmIDs": self.mmIDs
        }
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BOOK RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BOOK RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BOOK RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    @staticmethod
    def rawLoad(data: dict, bookID: str | None=None) -> 'Book':
        reqParams = ['title', 'subtitle', 'mmIDs']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'mmIDs':
                    data[reqParam] = []
                else:
                    data[reqParam] = None
        
        return Book(
            title=data.get('title'),
            subtitle=data.get('subtitle'),
            mmIDs=data.get('mmIDs', []),
            id=bookID
        )
    
    @staticmethod
    def load(id: str=None, title: str=None, withMMID: str=None) -> 'Book | List[Book] | None':
        """
        Loads a book(s) by ID, title, or MMData artefact ID.

        Args:
            id (str): Book ID.
            title (str): Book title to search for.
            withMMID (str, optional): MMData artefact ID to search in book's mmIDs.

        Returns:
            Book | List[Book] | None
        """
        if id != None:
            data = DI.load(Book.ref(id))
            if data is None:
                if title is not None or withMMID is not None:
                    return Book.load(title=title, withMMID=withMMID)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("BOOK LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BOOK LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            return Book.rawLoad(data, id)

        else:
            data = DI.load(Ref("books"))
            if data is None:
                return None
            if isinstance(data, DIError):
                raise Exception("BOOK LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BOOK LOAD ERROR: Failed to load dictionary books data; response: {}".format(data))

            books: Dict[str, Book] = {}
            for id in data:
                if isinstance(data[id], dict):
                    books[id] = Book.rawLoad(data[id], id)
            
            if title is None and withMMID is None:
                return list(books.values())
            
            for book in books.values():
                if title is not None and book.title == title:
                    return book
                elif withMMID is not None and withMMID in book.mmIDs:
                    return book

            return None

    @staticmethod
    def ref(id: str):
        return Ref("books", id)

class Batch(DIRepresentable):
    """A collection of references to artefacts that are processed together. A `DIRepresentable` model.
    
    Notes:
    - `Batch` is a larger model that has a collection of `BatchArtefact` objects, which are references to `Artefact` objects.
    - Processing job data associated with a batch is stored as `BatchProcessingJob` under the `job` attribute.
    - The memory size of this model can get quite extensive if all sub-units of data are loaded and there are many artefacts.
    - Each `BatchArtefact` has a `stage` attribute that indicates its processing status, among other parameters to store processing information.
    - Do note that `Batch.Stage` (at `Batch().stage`), `BatchArtefact.Status` (at `BatchArtefact().stage`) and `BatchProcessingJob.Status` (at `BatchProcessingJob().status`) are three semantically different status indicators.
    - Each `job: BatchProcessingJob` has a `status` attribute that indicates the status of the processing job. It also contains a `jobID`, which is meant to be the `ThreadManager` job ID.
    - Methods below are excluding those already defined in `DIRepresentable`.
    
    Attributes:
        id (str): Unique identifier for the batch.
        userID (str): ID of the user who created the batch.
        artefacts (Dict[str, BatchArtefact]): Dictionary of `BatchArtefact` objects keyed by artefact ID.
        job (BatchProcessingJob | None): The processing job associated with the batch, if any.
        stage (Batch.Stage): The stage that the batch is in.
        user (User | None): The user object associated with the batch, if loaded.
        created (str): Timestamp of when the batch was created.
        originRef (Ref): Reference to the batch in the database.
    
    Methods:
        `save(checkIntegrity: bool=True)`: Saves the batch to the database, checking integrity if specified.
        `represent()`: Returns a dictionary representation of the batch.
        `reload()`: Reloads the batch data from the database.
        `getUser()`: Loads the user associated with the batch.
        `loadAllArtefacts(withMetadata: bool=False)`: Loads all artefacts in the batch, optionally including metadata. Artefacts are loaded internally in the `BatchArtefact` objects (`artefact` attribute).
        `get(artefact: Artefact | str)`: Retrieves a `BatchArtefact` by its artefact ID or `Artefact` object.
        `add(artefact: Artefact | str, to: BatchArtefact.Status)`: Adds an artefact to the batch at a specified stage.
        `has(artefact: Artefact | str)`: Checks if an artefact is in the batch.
        `move(artefact: Artefact | str, to: BatchArtefact.Status)`: Moves an artefact to a different stage in the batch.
        `where(artefact: Artefact | str)`: Returns the stage of an artefact in the batch.
        `remove(artefact: Artefact | str)`: Removes an artefact from the batch.
        `initJob(jobID: str=None, status: BatchProcessingJob.Status=None, autoStart: bool=False)`: Initializes a processing job for the batch with an optional job ID and status, and optionally starts it.
        `__str__()`: Returns a string representation of the batch, including its ID, user ID, artefacts, job, and creation timestamp.
    
    Static Methods:
        `rawLoad(batchID: str, data: dict, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False)`: Loads a batch from raw data, optionally including user and artefact information.
        `load(id: str=None, userID: str=None, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False)`: Loads a batch by ID or user ID, optionally including user and artefact information.
        `ref(id: str)`: Returns a reference to the batch in the database.
    
    Example usage:
    ```python
    from schemas import DI, Batch, BatchArtefact, BatchProcessingJob, User, Artefact
    DI.setup()
    
    user = User.load()[0] # Load the first user available
    arts = Artefact.load()
    
    batch: Batch = Batch(user.id, arts) # Create a new batch, associated with the given user, containing the artefacts. Artefacts are automatically converted to BatchArtefact objects in the 'unprocessed' stage.
    
    batch.loadAllArtefacts(withMetadata=True) # Loads all artefacts in the batch, including their metadata if available. Warning: time- and memory-intensive operation.
    print(list(batch.artefacts.values())[0].artefact) # outputs the Artefact object of the first BatchArtefact reference in the batch
    
    print(batch.where(arts[0])) # outputs: BatchArtefact.Status.UNPROCESSED
    
    newArt = Artefact.load('some_artefact_not_in_batch')
    
    print(batch.has(newArt)) # outputs: False
    batch.add(newArt, BatchArtefact.Status.UNPROCESSED) # constructs BatchArtefact reference to new artefact with the 'unprocessed' stage
    print(batch.has(newArt)) # outputs: True

    print(batch.move(arts[0], BatchArtefact.Status.PROCESSED))
    print(batch.where(arts[0])) # outputs: BatchArtefact.Status.PROCESSED
    
    batch.initJob(jobID='someJobID', autoStart=True) # Initializes a processing job for the batch
    print(batch.job) # outputs BatchProcessingJob instance
    
    batch.job.end() # note: .cancel() auto saves by default
    batch.save() # remember to save after any modifications
    print(batch) # outputs the string representation of the batch with all artefacts and job information
    ```
    """
    
    class Stage(str, Enum):
        UPLOAD_PENDING = "upload_pending"
        UNPROCESSED = "unprocessed"
        PROCESSED = "processed"
        VETTING = "vetting"
        INTEGRATION = "integration"
        COMPLETED = "completed"

        @staticmethod
        def validateAndReturn(stage: 'Batch.Stage | str') -> 'Batch.Stage | Literal[False]':
            """Validates the given stage and returns it if valid, or False if invalid.

            Args:
                stage (str | Batch.Stage): The stage to validate.

            Returns: `Batch.Stage | Literal[False]`: The validated stage or False if invalid.
            """
            
            if isinstance(stage, Batch.Stage):
                return stage
            elif isinstance(stage, str):
                try:
                    return Batch.Stage(stage.lower())
                except ValueError:
                    return False
            else:
                return False

    def __init__(self, userID: str, name: str=None, batchArtefacts: 'List[BatchArtefact | Artefact]'=None, job: 'BatchProcessingJob | None'=None, stage: 'Batch.Stage'=None, created: str=None, id: str=None):
        """Initializes a new Batch instance.

        Args:
            userID (str): The ID of the user associated with the batch.
            name (str | None): The name of the batch, optional. Defaults to None.
            batchArtefacts (List[BatchArtefact | Artefact], optional): The artefacts to include in the batch. Both `BatchArtefact` and `Artefact` objects are accepted. `Artefact` objects are converted into 'unprocessed' `BatchArtefact` objects. Defaults to None.
            job (BatchProcessingJob | None, optional): The processing job associated with the batch. Defaults to None.
            stage (Batch.Stage, optional): The stage of the batch. Defaults to `Batch.Stage.UPLOAD_PENDING`.
            created (str, optional): The creation timestamp of the batch. Defaults to None.
            id (str, optional): The unique ID of the batch. Defaults to None.

        Raises:
            Exception: If the initialization fails or if the provided artefacts are not of the expected type.
        """
        
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if batchArtefacts is None:
            batchArtefacts = {}
        if stage is None:
            stage = Batch.Stage.UPLOAD_PENDING
        
        batchArts = {}
        for item in batchArtefacts:
            if isinstance(item, BatchArtefact):
                batchArts[item.artefactID] = item
            elif isinstance(item, Artefact):
                batchArts[item.id] = BatchArtefact(id, item.id, BatchArtefact.Status.UNPROCESSED)
            else:
                raise Exception("BATCH INIT ERROR: Expected item in 'batchArtefacts' to be either BatchArtefact or Artefact, but got '{}'.".format(type(item).__name__))
        
        self.id: str = id
        self.userID: str = userID
        self.name: str | None = name
        self.artefacts: Dict[str, BatchArtefact] = batchArts
        self.job: BatchProcessingJob | None = job
        self.stage: Batch.Stage = stage
        self.user: User | None = None
        self.created: str = created
        self.originRef = Batch.ref(self.id)
    
    def save(self, checkIntegrity: bool=True) -> bool:
        """Saves the batch to the database.

        Args:
            checkIntegrity (bool, optional): Whether to check the integrity of the batch before saving. Defaults to True.

        Raises:
            Exception: If the batch does not have a userID set.
            Exception: If the user associated with the batch does not exist.
            Exception: If any artefact in the batch is not a valid BatchArtefact.
            Exception: If the batch's job is not a valid BatchProcessingJob.
            Exception: If a batch artefact's batch ID does not match `self.batchID`.
            Exception: If a batch artefact's artefact ID does not match the expected ID.

        Returns:
            bool: True if the batch was saved successfully.
        """
        
        if checkIntegrity:
            if self.userID is None:
                raise Exception("BATCH SAVE ERROR: Batch does not have a userID set.")
            
            data = User.load(self.userID)
            if data is None:
                raise Exception("BATCH SAVE ERROR: User with ID '{}' does not exist.".format(self.userID))
            del data  # We don't need the user data here, just checking existence
            
            for artefactID, batchArt in self.artefacts.items():
                if not isinstance(batchArt, BatchArtefact):
                    raise Exception("BATCH SAVE ERROR: Artefact with ID '{}' is not a valid BatchArtefact.".format(artefactID))
                if batchArt.batchID != self.id:
                    raise Exception("BATCH SAVE ERROR: Artefact with ID '{}' does not belong to this batch (ID: '{}').".format(artefactID, self.id))
                if batchArt.artefactID != artefactID:
                    raise Exception("BATCH SAVE ERROR: Found artefact ID '{}' in BatchArtefact when '{}' was expected.".format(batchArt.artefactID, artefactID))
                if BatchArtefact.Status.validateAndReturn(batchArt.stage) == False:
                    raise Exception("BATCH SAVE ERROR: Invalid stage '{}' for artefact with ID '{}'.".format(batchArt.stage, artefactID))
            
            if self.job != None and not isinstance(self.job, BatchProcessingJob):
                raise Exception("BATCH SAVE ERROR: 'job' is neither None nor a valid BatchProcessingJob instance.")
        
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "userID": self.userID,
            "name": self.name,
            "artefacts": {id: batchArt.represent() for id, batchArt in self.artefacts.items()},
            'job': self.job.represent() if self.job != None else None,
            'stage': self.stage.value,
            "created": self.created
        }
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BATCH RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BATCH RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BATCH RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.id, data).__dict__)
        return True
    
    def __str__(self):
        return """<Batch instance
ID: {}
Name: {}
User ID: {}
User Object:{}
Artefact References:{}
Job:{}
Stage: {}
Created: {} />
        """.format(
            self.id,
            self.name,
            self.userID,
            self.user.represent() if isinstance(self.user, User) else " None",
            ("\n---\n- " + ("\n- ".join(str(batchArt) if isinstance(batchArt, BatchArtefact) else "CORRUPTED BATCHARTEFACT AT '{}'".format(artID) for artID, batchArt in self.artefacts.items())) + "\n---") if isinstance(self.artefacts, dict) else " None",
            self.job if isinstance(self.job, BatchProcessingJob) else " None",
            self.stage.value,
            self.created
        )
    
    def getUser(self) -> bool:
        """Loads the user associated with the batch into the `user` attribute.

        Raises:
            Exception: If the `userID` is invalid or if the user cannot be loaded.

        Returns:
            bool: True if the user was successfully loaded, False otherwise.
        """
        
        if not isinstance(self.userID, str):
            raise Exception("BATCH GETUSER ERROR: Invalid value set for 'userID'. Expected a string representing the User ID.")
        
        self.user = User.load(id=self.userID)
        if not isinstance(self.user, User):
            self.user = None
            raise Exception("BATCH GETUSER ERROR: Failed to load User with ID '{}'; response: {}".format(self.userID, self.user))
        
        return True
    
    def loadAllArtefacts(self, withMetadata: bool=False) -> bool:
        """Loads all artefacts associated with the batch.

        Args:
            withMetadata (bool, optional): If True, includes metadata in the artefact loading process. Defaults to False.

        Raises:
            Exception: If the `artefacts` attribute is not a dictionary.
            Exception: If a value in `artefacts` is not a valid `BatchArtefact`.
            Exception: If an artefact cannot be loaded.

        Returns:
            bool: True if all artefacts were successfully loaded.
        """
        
        if not isinstance(self.artefacts, dict):
            raise Exception("BATCH LOADARTEFACTS ERROR: Expected 'artefacts' to be a dictionary, but got '{}'.".format(type(self.artefacts).__name__))
        
        for artefactID, batchArt in self.artefacts.items():
            if not isinstance(batchArt, BatchArtefact):
                raise Exception("BATCH LOADARTEFACTS ERROR: Artefact with ID '{}' is not a valid BatchArtefact.".format(artefactID))
            
            batchArt.getArtefact(includeMetadata=withMetadata)
        
        return True
    
    def get(self, artefact: Artefact | str) -> 'BatchArtefact | None':
        """Retrieves a `BatchArtefact` reference from the batch.

        Args:
            artefact (Artefact | str): The artefact to retrieve, either as an Artefact instance or its ID.

        Returns: `BatchArtefact | None`: The requested artefact if it exists, None otherwise.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return self.artefacts.get(artefactID, None)
    
    def add(self, artefact: Artefact | str, to: 'BatchArtefact.Status') -> 'BatchArtefact':
        """Adds an artefact to the batch, creating a `BatchArtefact` reference.

        Args:
            artefact (Artefact | str): The artefact to add, either as an Artefact instance or its ID.
            to (BatchArtefact.Status): The stage of the artefact reference.

        Raises:
            Exception: If the `artefact` value is not valid.
            Exception: If the artefact reference already exists in the batch.

        Returns:
            BatchArtefact: The added artefact reference object.
        """
        
        to = BatchArtefact.Status.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH ADD ERROR: Invalid stage provided; expected 'BatchArtefact.Status' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if self.has(artefactID):
            raise Exception("BATCH ADD ERROR: Artefact with ID '{}' already exists in the batch.".format(artefactID))
        
        batchArt = BatchArtefact(self.id, artefactID, to)
        self.artefacts[artefactID] = batchArt
        
        return batchArt
    
    def has(self, artefact: Artefact | str) -> bool:
        """Checks if a reference to an artefact exists in the batch.

        Args:
            artefact (Artefact | str): The artefact to check, either as an Artefact instance or its ID.

        Returns:
            bool: True if the artefact exists in the batch, False otherwise.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return artefactID in self.artefacts
    
    def move(self, artefact: Artefact | str, to: 'BatchArtefact.Status') -> bool:
        """Moves an artefact reference to a different stage in the batch.

        Args:
            artefact (Artefact | str): The artefact to move, either as an Artefact instance or its ID.
            to (BatchArtefact.Status): The stage to which the artefact should be moved.

        Raises:
            Exception: If the `artefact` value is not valid.
            Exception: If the artefact reference does not exist in the batch.

        Returns:
            bool: True if the artefact was moved successfully, False otherwise.
        """
        
        to = BatchArtefact.Status.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH MOVE ERROR: Invalid stage provided; expected 'BatchArtefact.Status' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if not self.has(artefactID):
            raise Exception("BATCH MOVE ERROR: Artefact with ID '{}' does not exist in the batch.".format(artefactID))
        
        batchArt = self.artefacts[artefactID]
        batchArt.stage = to
        
        return True
    
    def where(self, artefact: Artefact | str) -> 'BatchArtefact.Status | None':
        """Retrieves the stage of a specific artefact reference in the batch.

        Args:
            artefact (Artefact | str): The artefact to check, either as an Artefact instance or its ID.

        Returns: `BatchArtefact.Status | None`: The stage of the artefact if it exists in the batch, None otherwise.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return self.artefacts[artefactID].stage if self.has(artefactID) else None
    
    def remove(self, artefact: Artefact | str) -> bool:
        """Removes an artefact reference from the batch.

        Args:
            artefact (Artefact | str): The artefact to remove, either as an Artefact instance or its ID.

        Returns:
            bool: True if the artefact was removed successfully.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        self.artefacts.pop(artefactID, None)
        return True
    
    def initJob(self, jobID: str=None, status: 'BatchProcessingJob.Status'=None, autoStart: bool=False) -> 'BatchProcessingJob':
        """Initializes a new `BatchProcessingJob` under the `job` attribute.

        Args:
            jobID (str, optional): The ID of the job. Defaults to None.
            status (BatchProcessingJob.Status, optional): The status of the job. Defaults to `BatchProcessingJob.Status.PENDING`.
            autoStart (bool, optional): Whether to start the job automatically. Defaults to False.

        Raises:
            Exception: If the `status` value is provided but not valid.

        Returns:
            BatchProcessingJob: The initialized batch processing job object.
        """
        
        if status is None:
            status = BatchProcessingJob.Status.PENDING
        
        status = BatchProcessingJob.Status.validateAndReturn(status)
        if status == False:
            raise Exception("BATCH INITJOB ERROR: Invalid status provided; expected 'BatchProcessingJob.Status' or string representation of it.")
        
        if isinstance(self.job, BatchProcessingJob):
            if self.job.status == BatchProcessingJob.Status.PROCESSING or self.job.ended is None:
                Logger.log("BATCH INITJOB WARNING: Overwriting existing job with processing status or no 'ended' value with new job with ID: '{}' and status: '{}'.".format(jobID, status))
        
        self.job = BatchProcessingJob(self.id, jobID, status)
        
        if autoStart:
            self.job.start(jobID)
        
        return self.job
    
    def __str__(self):
        return """<Batch instance
ID: {}
User ID: {}
User Object: {}
Artefacts:{}
Job: {}
Created: {} />
        """.format(
            self.id,
            self.userID,
            self.user.represent() if self.user is not None else " None",
            ("\n---\n- " + ("\n- ".join(str(batchArt) if isinstance(batchArt, BatchArtefact) else "CORRUPTED BATCHARTEFACT AT '{}'".format(artID) for artID, batchArt in self.artefacts.items())) + "\n---") if isinstance(self.artefacts, dict) else " None",
            self.job,
            self.created
        )
    
    @staticmethod
    def rawLoad(batchID: str, data: dict, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Batch':
        """Constructs a `Batch` from raw dictionary data.

        Args:
            batchID (str): The ID of the batch.
            data (dict): The raw data for the batch.
            withUser (bool, optional): Whether to automatically load the associated `User` object. Defaults to False.
            withArtefacts (bool, optional): Whether to automatically load `Artefact` objects internally in `BatchArtefact` references. Defaults to False.
            metadataRequired (bool, optional): Whether metadata should be loaded in the Artefact loading process. Defaults to False. Effective if `withArtefacts` is True. Warning: may cause high memory usage.

        Returns:
            Batch: The loaded batch object.
        """
        
        requiredParams = ['userID', 'name', 'artefacts', 'job', 'stage', 'created']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam == 'artefacts':
                    data[reqParam] = {}
                else:
                    data[reqParam] = None
        
        if not isinstance(data['artefacts'], dict):
            data['artefacts'] = {}
        
        arts = []
        for artefactID, batchArtData in data['artefacts'].items():
            if isinstance(batchArtData, dict):
                arts.append(BatchArtefact.rawLoad(batchID, artefactID, batchArtData))
        
        job = None
        if isinstance(data['job'], dict):
            job = BatchProcessingJob.rawLoad(batchID, data['job'])
        
        stage: Batch.Stage | None = None
        if data['stage']:
            stage = Batch.Stage.validateAndReturn(data['stage'])
            if stage == False:
                stage = None
        
        batch = Batch(
            userID=data.get('userID'),
            name=data.get('name'),
            batchArtefacts=arts,
            job=job,
            stage=stage,
            created=data.get('created'),
            id=batchID
        )
        
        if withUser:
            batch.getUser()
        if withArtefacts:
            batch.loadAllArtefacts(withMetadata=metadataRequired)
        
        return batch
    
    @staticmethod
    def load(id: str=None, userID: str=None, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Batch | Dict[Batch] | None':
        """Loads a batch from the database.

        Args:
            id (str, optional): The ID of the batch to load. Defaults to None.
            userID (str, optional): Produces a list of batches associated with the user ID. Defaults to None.
            withUser (bool, optional): Whether to automatically load the associated `User` object. Defaults to False.
            withArtefacts (bool, optional): Whether to automatically load `Artefact` objects internally in `BatchArtefact` references. Defaults to False.
            metadataRequired (bool, optional): Whether metadata should be loaded in the Artefact loading process. Defaults to False. Effective if `withArtefacts` is True. Warning: may cause high memory usage.

        Returns: `Batch | Dict[Batch] | None`: The loaded batch or list of batches, or None if not found.
        """
        
        if id != None:
            data = DI.load(Batch.ref(id))
            if data is None:
                if userID is not None:
                    return Batch.load(userID=userID, withUser=withUser, withArtefacts=withArtefacts, metadataRequired=metadataRequired)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("BATCH LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BATCH LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            batch = Batch.rawLoad(id, data, withUser, withArtefacts, metadataRequired)
            
            return batch
        else:
            data = DI.load(Ref("batches"))
            if data is None:
                return []
            if isinstance(data, DIError):
                raise Exception("BATCH LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BATCH LOAD ERROR: Failed to load dictionary batches data; response: {}".format(data))
            
            batches: Dict[str, Batch] = {}
            for id in data:
                if isinstance(data[id], dict):
                    batches[id] = Batch.rawLoad(id, data[id], withUser, withArtefacts, metadataRequired)
            
            if userID is None:
                return list(batches.values())
            
            userBatches: List[Batch] = []
            for batchID in batches:
                if batches[batchID].userID == userID:
                    userBatches.append(batches[batchID])
            
            return userBatches
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("batches", id)

class BatchProcessingJob(DIRepresentable):
    """Represents a batch processing job. Contains key attributes such as job ID, status and timestamps for when the job started and ended. A `DIRepresentable` model.

    Attributes:
        batchID (str): The ID of the batch this job is associated with.
        jobID (str | None): The ID of the job, if available.
        status (BatchProcessingJob.Status): The current status of the job.
        started (str | None): The timestamp when the job started, if available.
        ended (str | None): The timestamp when the job ended, if available.
    
    Methods:
        `save()`: Saves the job data to the database.
        `represent()`: Returns a dictionary representation of the job.
        `start(jobID: str=None)`: Starts the job, setting its status to PROCESSING and updating the started timestamp. Optionally sets a job ID.
        `end()`: Ends the job, setting its status to COMPLETED and updating the ended timestamp.
        `cancel(autoSave: bool=True)`: Cancels the job, setting its status to CANCELLED and updating the ended timestamp. Optionally saves the job data.
        `reload()`: Reloads the job data from the database.
        `setStatus(newStatus: BatchProcessingJob.Status)`: Sets the job's status to a new value, validating it first.
        `__str__()`: Returns a string representation of the job.
    
    Static Methods:
        `rawLoad(batchID: str, data: dict)`: Constructs a `BatchProcessingJob` from raw data, validating required parameters and setting defaults if necessary.
        `load(batch: Batch | str)`: Loads a `BatchProcessingJob` for a given batch, either by batch ID or `Batch` object. Returns None if no job is found.
        `ref(batchID: str)`: Returns a reference to the job in the database.
    
    Example usage:
    ```python
    import time
    from schemas import DI, Batch, BatchProcessingJob
    DI.setup()
    
    job = BatchProcessingJob.load(batchID='some_batch_id')
    job.start()
    
    time.sleep(2)  # Simulate processing time
    
    job.end()
    job.save()  # Save the job data to the database
    print(job)  # Outputs the string representation of the job
    ```
    """
    
    class Status(str, Enum):
        """Represents the various statuses a batch processing job can have.
        
        Options:
            PENDING: The job is waiting to be processed.
            PROCESSING: The job is currently being processed.
            COMPLETED: The job has been successfully completed.
            CANCELLED: The job has been cancelled.
        """
        
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        CANCELLED = "cancelled"
        
        @staticmethod
        def validateAndReturn(status) -> 'BatchProcessingJob.Status | Literal[False]':
            if isinstance(status, BatchProcessingJob.Status):
                return status
            elif isinstance(status, str):
                try:
                    return BatchProcessingJob.Status(status.lower())
                except ValueError:
                    return False
            else:
                return False
    
    def __init__(self, batch: Batch | str, jobID: str | None, status: 'BatchProcessingJob.Status', started: str | None=None, ended: str | None=None):
        """Initializes a new batch processing job.

        Args:
            batch (Batch | str): The batch object or ID associated with the job.
            jobID (str | None): Typically the thread ID of the job.
            status (BatchProcessingJob.Status): The current status of the job.
            started (str | None, optional): The timestamp when the job was started. Defaults to None.
            ended (str | None, optional): The timestamp when the job was ended. Defaults to None.

        Raises:
            Exception: If the provided status is invalid.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        status = BatchProcessingJob.Status.validateAndReturn(status)
        if status == False:
            raise Exception("BATCHPROCESSINGJOB INIT ERROR: Invalid status provided; expected 'BatchProcessingJob.Status' or string representation of it.")
        
        self.batchID: str = batchID
        self.jobID: str | None = jobID
        self.status: BatchProcessingJob.Status = status
        self.started: str | None = started
        self.ended: str | None = ended
        self.originRef = BatchProcessingJob.ref(batchID)
    
    def represent(self):
        return {
            "jobID": self.jobID,
            "status": self.status.value,
            "started": self.started,
            "ended": self.ended
        }
    
    def save(self):
        return DI.save(self.represent(), self.originRef)
    
    def start(self, jobID: str=None) -> bool:
        """Starts the batch processing job, setting its status to PROCESSING and updating the started timestamp.

        Args:
            jobID (str, optional): The unique identifier for the job. Defaults to None.

        Returns:
            bool: True if the job was successfully started.
        """
        
        if self.status != BatchProcessingJob.Status.PENDING:
            Logger.log("BATCHPROCESSINGJOB START WARNING: Starting job with ID '{}' that is not in PENDING status; current status: '{}'.".format(self.jobID, self.status.value))
        
        if self.jobID != None:
            self.jobID = jobID
        self.status = BatchProcessingJob.Status.PROCESSING
        self.started = Universal.utcNowString()
        
        return True
    
    def end(self) -> bool:
        """Ends the batch processing job, setting its status to COMPLETED and updating the ended timestamp.

        Returns:
            bool: True if the job was successfully ended.
        """
        
        if self.status != BatchProcessingJob.Status.PROCESSING:
            Logger.log("BATCHPROCESSINGJOB END WARNING: Ending job with ID '{}' that is not in PROCESSING status; current status: '{}'.".format(self.jobID, self.status.value))
        
        self.status = BatchProcessingJob.Status.COMPLETED
        self.ended = Universal.utcNowString()
        
        return True
    
    def cancel(self, autoSave: bool=True) -> bool:
        """Cancels the batch processing job, setting its status to CANCELLED and updating the ended timestamp.

        Args:
            autoSave (bool, optional): Whether to automatically save the job state. Defaults to True.

        Returns:
            bool: True if the job was successfully cancelled.
        """
        
        if self.status != BatchProcessingJob.Status.PROCESSING:
            Logger.log("BATCHPROCESSINGJOB CANCEL WARNING: Cancelling job with ID '{}' that is not in PROCESSING status; current status: '{}'.".format(self.jobID, self.status.value))
        
        self.status = BatchProcessingJob.Status.CANCELLED
        self.ended = Universal.utcNowString()
        
        if autoSave:
            return self.save()
        
        return True
    
    def fetchLatestStatus(self) -> 'BatchProcessingJob.Status | None':
        self.status = BatchProcessingJob.getStatus(self.batchID)
        return self.status
    
    def computeDuration(self) -> float | None:
        if not isinstance(self.started, str) or not isinstance(self.ended, str):
            return None
        try:
            start = Universal.fromUTC(self.started)
            end = Universal.fromUTC(self.ended)
            return (end - start).total_seconds()
        except Exception as e:
            Logger.log("BATCHPROCESSINGJOB COMPUTEDURATION ERROR: {}".format(e))
            return None
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BATCHPROCESSINGJOB RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BATCHPROCESSINGJOB RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BATCHPROCESSINGJOB RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))

        self.__dict__.update(self.rawLoad(self.batchID, data).__dict__)
        return True
    
    def setStatus(self, newStatus: 'BatchProcessingJob.Status') -> bool:
        """Sets the status of the batch processing job.

        Args:
            newStatus (BatchProcessingJob.Status): The new status to set.

        Raises:
            Exception: If the provided status is invalid.

        Returns:
            bool: True if the status was successfully updated.
        """
        
        newStatus = BatchProcessingJob.Status.validateAndReturn(newStatus)
        if newStatus == False:
            raise Exception("BATCHPROCESSINGJOB SETSTATUS ERROR: Invalid status provided; expected 'BatchProcessingJob.Status' or string representation of it.")
        
        self.status = newStatus
        return True
    
    def __str__(self):
        return "BatchProcessingJob(batchID='{}', jobID='{}', status='{}', started='{}', ended='{}')".format(
            self.batchID,
            self.jobID,
            self.status.value,
            self.started,
            self.ended
        )

    @staticmethod
    def rawLoad(batchID: str, data: dict) -> 'BatchProcessingJob':
        """Loads a batch processing job from the provided data dictionary.

        Args:
            batchID (str): The ID of the batch this job is associated with.
            data (dict): The data dictionary containing job information.

        Returns:
            BatchProcessingJob: The job object.
        """
        
        reqParams = ['jobID', 'status', 'started', 'ended']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'status':
                    data[reqParam] = BatchProcessingJob.Status.PENDING.value
                else:
                    data[reqParam] = None
        
        status = BatchProcessingJob.Status.validateAndReturn(data['status'])
        if status == False:
            status = BatchProcessingJob.Status.PENDING
        
        return BatchProcessingJob(
            batch=batchID,
            jobID=data.get('jobID'),
            status=status,
            started=data.get('started'),
            ended=data.get('ended')
        )
    
    @staticmethod
    def load(batch: Batch | str) -> 'BatchProcessingJob | None':
        """Loads a batch processing job from the provided batch identifier.

        Args:
            batch (Batch | str): The batch object or ID this job is associated with.

        Raises:
            Exception: If the provided batch is invalid.
            Exception: If the job cannot be loaded.

        Returns: `BatchProcessingJob | None`: The loaded job object or None if not found.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        
        data = DI.load(BatchProcessingJob.ref(batchID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHPROCESSINGJOB LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("BATCHPROCESSINGJOB LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return BatchProcessingJob.rawLoad(batchID, data)
    
    @staticmethod
    def getStatus(batch: Batch | str) -> 'BatchProcessingJob.Status | None':
        """Gets the status of the batch processing job.

        Args:
            batch (Batch | str): The batch object or ID to retrieve the job status for.

        Raises:
            Exception: If any error occurs during status retrieval and validation.

        Returns: `BatchProcessingJob.Status | None`: The status of the job if it exists, None otherwise.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        if not isinstance(batchID, str):
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: Expected 'batch' to be a Batch object or a string representing the batch ID, but got '{}'.".format(type(batch).__name__))
        
        data = DI.load(Ref("batches", batchID, "job", "status"))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, str):
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: Unexpected DI load response format; response: {}".format(data))
        
        status = BatchProcessingJob.Status.validateAndReturn(data)
        if status == False:
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: Invalid status value '{}' returned from DI.".format(data))
        
        return status
    
    @staticmethod
    def ref(batchID: str) -> Ref:
        return Ref("batches", batchID, "job")

class BatchArtefact(DIRepresentable):
    """Represents an artefact reference within a batch. A junction model between `Batch` and `Artefact`, this model contains key attributes
    that link the two entities together with additional information.
    
    Attributes:
        batchID (str): The ID of the batch this artefact is associated with.
        artefactID (str): The ID of the artefact this reference points to.
        stage (BatchArtefact.Status): The current processing stage of the artefact within the batch.
        processedDuration (str | None): The duration for which the artefact has been processed, if applicable.
        processedTime (str | None): The timestamp when the artefact was processed, if applicable.
        processingError (str | None): Any error that occurred during the processing of the artefact, if applicable.
        integrationResult (dict | None): Any information about the results of the artefact's catalogue integration procedure.
        artefact (Artefact | None): A convenience attribute that holds the `Artefact` object if it has been loaded.
        originRef (Ref): The reference to the artefact in the database.
    
    Methods:
        `save()`: Saves the artefact reference to the database.
        `represent()`: Returns a dictionary representation of the artefact reference.
        `reload()`: Reloads the artefact reference from the database.
        `getArtefact(includeMetadata: bool=False)`: Loads the associated `Artefact` object into the `artefact` attribute, optionally including metadata.
        `setStage(newStage: BatchArtefact.Status)`: Sets the stage of the artefact reference to a new value, validating it first
        `__str__()`: Returns a string representation of the artefact reference.
    
    Static Methods:
        `rawLoad(batchID: str, artefactID: str, data: dict, withArtefact: bool=False, metadataRequired: bool=False)`: Constructs a `BatchArtefact` from raw data, validating required parameters and setting defaults if necessary.
        `load(batch: Batch | str, artefact: Artefact | str, withArtefact: bool=False, metadataRequired: bool=False)`: Loads a `BatchArtefact` for a given batch and artefact, either by IDs or objects. Returns None if not found.
        `ref(batchID: str, artefactID: str)`: Returns a reference to the artefact in the database.
    
    Example usage:
    ```python
    from services import Universal
    from schemas import DI, Batch, Artefact, BatchArtefact
    DI.setup()
    
    batch = Batch.load(id='some_batch_id')
    artefact = Artefact.load(id='some_artefact_id')
    
    batchArtefact = BatchArtefact(batch=batch, artefact=artefact, stage=BatchArtefact.Status.PROCESSED)
    
    batchArtefact.processedDuration = "32.456"
    batchArtefact.processedTime = Universal.utcNowString()
    batchArtefact.save() # Save the artefact reference to the database
    
    print(batchArtefact)  # Outputs the string representation of the artefact reference
    ```
    """

    class Status(str, Enum):
        """Represents the various stages a batch artefact can be in.
        
        Options:
            UNPROCESSED: The artefact has not been processed yet.
            PROCESSED: The artefact has been processed.
            CONFIRMED: The artefact has been confirmed after processing.
            INTEGRATED: The artefact has been integrated into the catalogue.
        """
        
        UNPROCESSED = "unprocessed"
        PROCESSED = "processed"
        CONFIRMED = "confirmed"
        INTEGRATED = "integrated"
        
        @staticmethod
        def validateAndReturn(status: 'BatchArtefact.Status | str') -> 'BatchArtefact.Status | Literal[False]':
            """Validates the given stage and returns it if valid, or False if invalid.

            Args:
                stage (str | BatchArtefact.Status): The stage to validate.

            Returns: `BatchArtefact.Status | Literal[False]`: The validated stage or False if invalid.
            """
            
            if isinstance(status, BatchArtefact.Status):
                return status
            elif isinstance(status, str):
                try:
                    return BatchArtefact.Status(status.lower())
                except ValueError:
                    return False
            else:
                return False
    
    def __init__(self, batch: Batch | str, artefact: Artefact | str, stage: 'BatchArtefact.Status', processedDuration: str | None=None, processedTime: str | None=None, processingError: str | None=None, integrationResult: dict | None=None):
        """Initializes a BatchArtefact instance.

        Args:
            batch (Batch | str): The batch object or ID this artefact is associated with.
            artefact (Artefact | str): The artefact object or ID this reference is associated with.
            stage (BatchArtefact.Status): The current processing stage of the artefact.
            processedDuration (str | None, optional): The duration for which the artefact has been processed, if applicable. Defaults to None.
            processedTime (str | None, optional): The timestamp when the artefact was processed, if applicable. Defaults to None.
            processingError (str | None, optional): Any error that occurred during the processing of the artefact, if applicable. Defaults to None.
            integrationResult (dict | None): Any information about the results of the artefact's catalogue integration procedure. Defaults to None.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        stage = BatchArtefact.Status.validateAndReturn(stage)
        if stage == False:
            raise Exception("BATCHARTEFACT INIT ERROR: Invalid stage provided; expected 'BatchArtefact.Status' or string representation of it.")
        
        self.batchID: str = batchID
        self.artefactID: str = artefactID
        self.stage: BatchArtefact.Status = stage
        self.processedDuration: str | None = processedDuration
        self.processedTime: str | None = processedTime
        self.processingError: str | None = processingError
        self.integrationResult: dict | None = integrationResult
        self.artefact: Artefact | None = artefact if isinstance(artefact, Artefact) else None
        self.originRef = BatchArtefact.ref(batchID, artefactID)
    
    def represent(self):
        return {
            "stage": self.stage.value,
            "processedDuration": self.processedDuration,
            "processedTime": self.processedTime,
            "processingError": self.processingError,
            "integrationResult": self.integrationResult
        }
    
    def save(self):
        return DI.save(self.represent(), self.originRef)
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BATCHARTEFACT RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BATCHARTEFACT RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BATCHARTEFACT RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.batchID, self.artefactID, data).__dict__)
        return True
    
    def getArtefact(self, includeMetadata: bool=False) -> bool:
        """Retrieves the associated `Artefact` object into `artefact` for convenience.

        Args:
            includeMetadata (bool, optional): Whether to include metadata in the retrieval. Defaults to False.

        Raises:
            Exception: If the `artefactID` is not a string or if the artefact cannot be loaded.

        Returns:
            bool: True if the artefact was successfully retrieved.
        """
        
        if not isinstance(self.artefactID, str):
            raise Exception("BATCHARTEFACT GETARTEFACT ERROR: Invalid value set for 'artefactID'. Expected a string representing the Artefact ID.")
        
        self.artefact = Artefact.load(id=self.artefactID, includeMetadata=includeMetadata)
        if not isinstance(self.artefact, Artefact):
            self.artefact = None
            raise Exception("BATCHARTEFACT GETARTEFACT ERROR: Failed to load Artefact with ID '{}'; response: {}".format(self.artefactID, self.artefact))
        
        return True
    
    def setStage(self, newStage: 'BatchArtefact.Status') -> bool:
        """Sets the processing stage of the artefact reference.

        Args:
            newStage (BatchArtefact.Status): The new stage to set.

        Raises:
            Exception: If the provided stage is invalid.

        Returns:
            bool: True if the stage was successfully set.
        """
        
        newStage = BatchArtefact.Status.validateAndReturn(newStage)
        if newStage == False:
            raise Exception("BATCHARTEFACT SETSTAGE ERROR: Invalid stage provided; expected 'BatchArtefact.Status' or string representation of it.")
        
        self.stage = newStage
        return True
    
    def __str__(self):
        return "BatchArtefact(batchID='{}', artefactID='{}', stage='{}', processedDuration='{}', processedTime='{}', processingError='{}', integrationResult='{}', artefact={})".format(
            self.batchID,
            self.artefactID,
            self.stage.value,
            self.processedDuration,
            self.processedTime,
            self.processingError,
            self.integrationResult,
            self.artefact if self.artefact != None else None
        )
    
    @staticmethod
    def rawLoad(batchID: str, artefactID: str, data: dict, withArtefact: bool=False, metadataRequired: bool=False) -> 'BatchArtefact':
        """Loads a BatchArtefact from raw dictionary data.

        Args:
            batchID (str): The ID of the batch this artefact is associated with.
            artefactID (str): The ID of the referred artefact.
            data (dict): The raw batch artefact reference information.
            withArtefact (bool, optional): Whether to automatically load the `Artefact` object internally. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is included in the `Artefact` loading process. Defaults to False. Effective if `withArtefact` is True. Warning: may cause high memory usage.

        Returns:
            BatchArtefact: The loaded BatchArtefact instance.
        """
        
        reqParams = ['stage', 'processedDuration', 'processedTime', 'processingError', "integrationResult"]
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'stage':
                    data[reqParam] = BatchArtefact.Status.UNPROCESSED.value
                else:
                    data[reqParam] = None
        
        stage = BatchArtefact.Status.validateAndReturn(data['stage'])
        if stage == False:
            stage = BatchArtefact.Status.UNPROCESSED
        
        batchArt = BatchArtefact(
            batch=batchID,
            artefact=artefactID,
            stage=stage,
            processedDuration=data.get('processedDuration'),
            processedTime=data.get('processedTime'),
            processingError=data.get('processingError'),
            integrationResult=data.get('integrationResult'),
        )
        
        if withArtefact:
            batchArt.getArtefact(includeMetadata=metadataRequired)
        
        return batchArt
    
    @staticmethod
    def load(batch: Batch | str, artefact: Artefact | str, withArtefact: bool=False, metadataRequired: bool=False) -> 'BatchArtefact | None':
        """Loads a BatchArtefact instance.

        Args:
            batch (Batch | str): The batch object or ID associated with the artefact reference.
            artefact (Artefact | str): The artefact object or ID associated with the artefact reference.
            withArtefact (bool, optional): Whether to automatically load the `Artefact` object internally. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is included in the `Artefact` loading process. Defaults to False. Effective if `withArtefact` is True. Warning: may cause high memory usage.

        Returns: `BatchArtefact | None`: The loaded BatchArtefact instance or None if not found.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        data = DI.load(BatchArtefact.ref(batchID, artefactID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("BATCHARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return BatchArtefact.rawLoad(batchID, artefactID, data, withArtefact, metadataRequired)
    
    @staticmethod
    def ref(batchID: str, artefactID: str) -> Ref:
        return Ref("batches", batchID, "artefacts", artefactID)

class Category(DIRepresentable):
    """
    Represents a collection of artefacts grouped under a specific category.

    Attributes:
        id (str): Unique identifier for the category.
        name (str): Name of the category.
        description (str): Description of the category.
        members (Dict[str, CategoryArtefact]): Dictionary mapping artefact IDs to CategoryArtefact objects.
        created (str): ISO timestamp of category creation.
        originRef (Ref): Reference object for database operations.

    Methods:
        represent(): Returns a dictionary representation of the category for serialization.
        save(): Saves the category to the database.
        reload(): Reloads the category data from the database.
        add(): Adds an artefact to the category.
        remove(): Removes an artefact from the category.
        has(): Checks if an artefact is in the category.
        get(): Retrieves the CategoryArtefact for a given artefact.
        cleanMembers(): Removes invalid or missing artefacts from members in a safe manner.
        loadAllArtefacts(): Loads all artefact objects for the category.
        rawLoad(): Loads a Category from raw dictionary data.
        load(): Loads a category by ID, name, or member ID.
        ref(id): Returns the database reference for the category.

    Sample usage:
    ```python
    from schemas import DI, Category, Artefact
    DI.setup()

    # Create a new category
    cat = Category(name="Paintings", description="All painting artefacts")
    cat.save()

    # Add an artefact to the category
    artefact = Artefact.load(id="artefact123")
    cat.add(artefact, reason="Featured in exhibition")
    cat.save()

    # Load and inspect a category
    loaded_cat = Category.load(name="Paintings", withArtefacts=True, metadataRequired=True) # loads 'Artefact' objects, which internally loads 'Metadata', internally in 'CategoryArtefact'
    print(loaded_cat.represent())
    ```

    Notes:
        - All artefacts added to a category must exist in the database.
        - The `members` attribute is a dictionary mapping artefact IDs to CategoryArtefact objects.
        - Use `save()` after modifying a category to persist changes.
        - The class supports both direct artefact objects and artefact IDs for member management.
    """
    
    def __init__(self, name: str, description: str, members: 'Dict[str, CategoryArtefact]'=None, created: str=None, id: str=None):
        """Initialize a new Category instance.

        Args:
            name (str): The name of the category.
            description (str): A brief description of the category.
            members (Dict[str, CategoryArtefact], optional): A dictionary of artefacts belonging to the category. Defaults to None.
            created (str, optional): ISO timestamp of category creation. Defaults to None.
            id (str, optional): Unique identifier for the category. Defaults to None.
        """
        
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if members is None:
            members = {}
        
        self.id = id
        self.name = name
        self.description = description
        self.members = members
        self.created = created
        self.originRef = Category.ref(id)
    
    def represent(self) -> dict:
        """Returns a dictionary representation of the category for serialization.

        Returns:
            dict: A dictionary containing the category's attributes.
        """
        
        return {
            "name": self.name,
            "description": self.description,
            "members": {artefactID: member.represent() for artefactID, member in self.members.items()},
            "created": self.created
        }
    
    def save(self, checkIntegrity: bool=True):
        """Saves the category to the database. Integrity check enforces unique names and integrity of 'members' attribute.
        
        Args:
            checkIntegrity (bool, optional): Whether to perform integrity checks before saving. Defaults to True.

        Returns:
            bool: True if the save operation was successful, False otherwise.
        """
        
        if checkIntegrity:
            if not isinstance(self.members, dict):
                raise Exception("CATEGORY SAVE ERROR: 'members' must be a dictionary of CategoryArtefact objects.")
            
            for artefactID, member in self.members.items():
                if not isinstance(member, CategoryArtefact):
                    raise Exception("CATEGORY SAVE ERROR: 'members' must contain CategoryArtefact objects; found type '{}' for artefact ID '{}'.".format(type(member), artefactID))
                if member.categoryID != self.id:
                    raise Exception("CATEGORY SAVE ERROR: CategoryArtefact with ID '{}' does not belong to this category (expected category ID '{}').".format(artefactID, self.id))
            
            cat = Category.load(name=self.name)
            if isinstance(cat, Category) and cat.id != self.id:
                raise Exception("CATEGORY SAVE ERROR: 'name' attribute uniqueness constraint fails due to category with ID '{}' with matching name '{}'.".format(cat.id, cat.name))

        return DI.save(self.represent(), self.originRef)
    
    def reload(self):
        """Reloads the category data from the database.

        Raises:
            Exception: If a DIError occurs during loading.
            Exception: If no data is found for the category.
            Exception: If the data format is unexpected.

        Returns:
            bool: True if the reload operation was successful, False otherwise.
        """
        
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("CATEGORY RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("CATEGORY RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("CATEGORY RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    def add(self, artefact: Artefact | str, reason: str, loadArtefactObject: bool=False, metadataRequired: bool=False) -> bool:
        """Adds an artefact to the category.

        Args:
            artefact (Artefact | str): The artefact to add, either as an Artefact object or a string ID.
            reason (str): The reason for adding the artefact.
            checkIntegrity (bool, optional): Whether to check the integrity of the artefact. Defaults to True.
            loadArtefactObject (bool, optional): Whether to load the artefact object. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is required. Defaults to False.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.
            Exception: If 'reason' is not a string.
            Exception: If the artefact already exists in the category.
            Exception: If the artefact cannot be loaded when checkIntegrity is True.
            Exception: If an artefact does not exist when loadArtefactObject is True.

        Returns:
            bool: True if the artefact was added successfully, False otherwise.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY ADD ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        if not isinstance(reason, str):
            raise Exception("CATEGORY ADD ERROR: 'reason' must be a string.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        if artefactID in self.members:
            raise Exception("CATEGORY ADD ERROR: Artefact with ID '{}' already exists.".format(artefactID))
        
        catArt = CategoryArtefact(
            category=self.id,
            artefact=artefact,
            reason=reason
        )
        
        if loadArtefactObject:
            catArt.getArtefact(includeMetadata=metadataRequired)
        
        self.members[artefactID] = catArt
        
        return True
    
    def remove(self, artefact: Artefact | str) -> bool:
        """Removes an artefact from the category.

        Args:
            artefact (Artefact | str): The artefact to remove, either as an Artefact object or a string ID.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.
            Exception: If the artefact does not exist in the category.

        Returns:
            bool: True if the artefact was removed successfully.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY REMOVE ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        if artefactID not in self.members:
            raise Exception("CATEGORY REMOVE ERROR: Artefact with ID '{}' does not exist in the category.".format(artefactID))
        
        del self.members[artefactID]
        return True

    def has(self, artefact: Artefact | str) -> bool:
        """Checks if the category contains a specific artefact.

        Args:
            artefact (Artefact | str): The artefact to check, either as an Artefact object or a string ID.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.

        Returns:
            bool: True if the artefact is in the category, False otherwise.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY HAS ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        return artefactID in self.members
    
    def get(self, artefact: Artefact | str, default: None=None) -> 'CategoryArtefact | None':
        """Retrieves an artefact from the category.

        Args:
            artefact (Artefact | str): The artefact to retrieve, either as an Artefact object or a string ID.
            default (None, optional): The value to return if the artefact is not found. Defaults to None.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.

        Returns:
            CategoryArtefact | None: The requested artefact, or the default value if not found.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY GET ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        return self.members.get(artefactID, default)
    
    def cleanMembers(self) -> bool:
        """Cleans up the members of the category by removing any invalid artefacts.

        Returns:
            bool: True if the cleanup was successful.
        """
        
        for artefactID in list(self.members.keys()):
            if not isinstance(self.members[artefactID], CategoryArtefact):
                del self.members[artefactID]
                continue
            
            if not self.members[artefactID].getArtefact(safeFail=True):
                del self.members[artefactID]
            
        return True
    
    def loadAllArtefacts(self, metadataRequired: bool=False) -> bool:
        """Loads all artefacts in the category.

        Args:
            metadataRequired (bool, optional): Whether to include metadata in the loading process. Defaults to False.

        Raises:
            Exception: If 'self.members' is not a dictionary.
            Exception: If any member is not a CategoryArtefact.

        Returns:
            bool: True if all artefacts were loaded successfully.
        """
        
        if not isinstance(self.members, dict):
            raise Exception("CATEGORY LOADALLARTEFACTS ERROR: 'members' must be a dictionary of CategoryArtefact objects.")
        
        for artefactID in self.members:
            if isinstance(self.members[artefactID], CategoryArtefact):
                catArt = self.members[artefactID]
                catArt.getArtefact(includeMetadata=metadataRequired)
            else:
                raise Exception("CATEGORY LOADALLARTEFACTS ERROR: Member with ID '{}' is not a CategoryArtefact.".format(artefactID))
        
        return True
    
    def __str__(self):
        return """<Category instance
ID: {}
Name: {}
Description: {}
Members:{}
Created: {} />    
        """.format(
            self.id,
            self.name,
            self.description,
            ("\n---\n- " + ("\n- ".join(str(member) if isinstance(member, CategoryArtefact) else "CORRUPTED MEMBER AT '{}'".format(member) for member in self.members.values())) + "\n---") if isinstance(self.members, dict) else " None",
            self.created
        )
    
    @staticmethod
    def rawLoad(data: dict, categoryID: str | None=None, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Category':
        """Loads a category from raw data.

        Args:
            data (dict): The raw data to load the category from.
            categoryID (str | None, optional): The ID of the category. Defaults to None.
            withArtefacts (bool, optional): Whether to include artefacts in the loading process. Defaults to False.
            metadataRequired (bool, optional): Whether to include metadata in the Artefact loading process. Defaults to False (memory constraints).

        Returns:
            Category: The loaded category.
        """
        
        reqParam = ['name', 'description', 'members', 'created']
        for param in reqParam:
            if param not in data:
                if param == 'members':
                    data[param] = {}
                else:
                    data[param] = None
        
        cat = Category(
            name=data.get('name'),
            description=data.get('description'),
            members=None,
            created=data.get('created'),
            id=categoryID
        )
        
        members = {}
        if isinstance(data['members'], dict):
            for artefactID in data['members']:
                if isinstance(data['members'][artefactID], dict):
                    members[artefactID] = CategoryArtefact.rawLoad(
                        category=cat.id,
                        artefact=artefactID,
                        data=data['members'][artefactID],
                        withArtefact=withArtefacts,
                        metadataRequired=metadataRequired
                    )
        
        cat.members = members
        return cat
    
    @staticmethod
    def load(id: str=None, name: str=None, withMemberID: str=None, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Category | List[Category] | None':
        """Loads a category from the data source.

        Args:
            id (str, optional): Retrieve by ID of the category. Defaults to None.
            name (str, optional): Retrieve by the name of the category. Defaults to None.
            withMemberID (str, optional): Retrieve by a category that has a certain member. Defaults to None.
            withArtefacts (bool, optional): Whether to include artefacts in the loading process. Defaults to False.
            metadataRequired (bool, optional): Whether to include metadata in the Artefact loading process. Defaults to False (memory constraints).

        Raises:
            Exception: If an error occurs during loading.
            Exception: If a DIError occurs during loading.
            Exception: If the data format is unexpected.

        Returns:
            Category | List[Category] | None: The loaded category or categories.
        """
        
        if id != None:
            data = DI.load(Category.ref(id))
            if data is None:
                if name is not None or withMemberID is not None:
                    return Category.load(name=name, withMemberID=withMemberID, withArtefacts=withArtefacts, metadataRequired=metadataRequired)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("CATEGORY LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("CATEGORY LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            return Category.rawLoad(data, id, withArtefacts=withArtefacts, metadataRequired=metadataRequired)
        else:
            data = DI.load(Ref("categories"))
            if data is None:
                return []
            if isinstance(data, DIError):
                raise Exception("CATEGORY LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("CATEGORY LOAD ERROR: Failed to load dictionary categories data; response: {}".format(data))

            categories: Dict[str, Category] = {}
            for id in data:
                if isinstance(data[id], dict):
                    categories[id] = Category.rawLoad(data[id], id, withArtefacts=withArtefacts, metadataRequired=metadataRequired)

            if name is None and withMemberID is None:
                return list(categories.values())

            for category in categories.values():
                if name is not None and category.name == name:
                    return category
                elif withMemberID is not None and withMemberID in category.members:
                    return category
            
            return None
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("categories", id)

class CategoryArtefact(DIRepresentable):
    """
    Represents the relationship between a Category and an Artefact, including metadata about the association.
    This can be considered similar to a join table in relational databases, i.e a *..* relationship between Category and Artefact.

    Attributes:
        categoryID (str): ID of the category this artefact belongs to.
        artefactID (str): ID of the artefact.
        reason (str): Reason for including the artefact in the category.
        added (str): ISO timestamp when the artefact was added to the category.
        artefact (Artefact | None): The loaded Artefact object, if available.
        originRef (Ref): Reference object for database operations.

    Methods:
        getArtefact(): Loads the Artefact object for this CategoryArtefact.
        represent(): Returns a dictionary representation for serialization.
        save(): Saves the CategoryArtefact to the database.
        rawLoad(): Loads a CategoryArtefact from raw dictionary data.
        load(): Loads a CategoryArtefact by category and artefact.
        ref(): Returns the database reference for the CategoryArtefact.

    Sample usage:
    ```python
    from schemas import DI, Category, CategoryArtefact, Artefact
    DI.setup()

    # Load a category and its artefact relationship
    cat = Category.load(name="Paintings")
    catArt = cat.get("artefact123") # catArt is of type CategoryArtefact
    print(catArt.represent())

    # Load the full artefact object
    catArt.getArtefact(includeMetadata=True) # loads the Artefact object, passing includeMetadata=True to the Artefact.load() method
    print(catArt.artefact.represent())
    ```

    Notes:
        - The `reason` attribute should describe why the artefact is included in the category.
        - Use `getArtefact()` to load the full Artefact object if needed.
        - Changes to CategoryArtefact objects should be saved using `save()`.
    """
    
    def __init__(self, category: Category | str, artefact: Artefact | str, reason: str, added: str=None):
        """Initializes a CategoryArtefact instance.

        Args:
            category (Category | str): The category this artefact belongs to.
            artefact (Artefact | str): The artefact object or its ID.
            reason (str): The reason for including the artefact in the category.
            added (str, optional): ISO timestamp when the artefact was added to the category. Defaults to UTC now timestamp.
        """
        
        if added is None:
            added = Universal.utcNowString()
        
        self.categoryID = category.id if isinstance(category, Category) else category
        self.artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        self.reason = reason
        self.added = added
        self.artefact: Artefact = artefact if isinstance(artefact, Artefact) else None
        self.originRef = CategoryArtefact.ref(self.categoryID, self.artefactID)
    
    def getArtefact(self, includeMetadata: bool=False, safeFail: bool=False) -> bool:
        """Loads the Artefact object for this CategoryArtefact.

        Args:
            includeMetadata (bool, optional): Whether to include metadata when loading the Artefact. Defaults to False.
            safeFail (bool, optional): Whether to fail silently if the Artefact cannot be loaded. Defaults to False.

        Raises:
            Exception: If the artefactID is not a string.
            Exception: If the Artefact cannot be loaded and safeFail is False.

        Returns:
            bool: True if the Artefact was loaded successfully, False otherwise (False is returned only if safeFail=True).
        """
        
        if not isinstance(self.artefactID, str):
            raise Exception("CATEGORYARTEFACT GETARTEFACT ERROR: Invalid value set for 'artefactID'. Expected a string representing the Artefact ID.")
        
        self.artefact = Artefact.load(self.artefactID, includeMetadata=includeMetadata)
        if not isinstance(self.artefact, Artefact):
            if safeFail:
                self.artefact = None
                return False
            raise Exception("CATEGORYARTEFACT GETARTEFACT ERROR: Failed to load Artefact with ID '{}'; response: {}".format(self.artefactID, self.artefact))
        
        return True
    
    def represent(self) -> dict:
        """Represents the CategoryArtefact instance as a dictionary.

        Returns:
            dict: The dictionary representation of the CategoryArtefact.
        """
        
        return {
            "reason": self.reason,
            "added": self.added
        }
    
    def save(self):
        """Saves the CategoryArtefact instance.

        Returns:
            bool: True if the save operation was successful.
        """
        
        return DI.save(self.represent(), self.originRef)
    
    def __str__(self):
        return "CategoryArtefact(categoryID='{}', artefactID='{}', reason='{}', added='{}', artefactObject={})".format(
            self.categoryID,
            self.artefactID,
            self.reason,
            self.added,
            self.artefact if isinstance(self.artefact, Artefact) else "None"
        )
    
    @staticmethod
    def rawLoad(category: Category | str, artefact: Artefact | str, data: dict, withArtefact: bool=False, metadataRequired: bool=False) -> 'CategoryArtefact':
        """Loads a CategoryArtefact instance from raw data.

        Args:
            category (Category | str): The category this artefact belongs to.
            artefact (Artefact | str): The artefact object or its ID.
            data (dict): The raw data to load from.
            withArtefact (bool, optional): Whether to load the Artefact object when loading. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is required when loading the Artefact. Defaults to False.

        Raises:
            Exception: If the categoryID is not a string.
            Exception: If the artefactID is not a string.

        Returns:
            CategoryArtefact: The loaded CategoryArtefact instance.
        """
        
        reqParams = ['reason', 'added']
        for reqParam in reqParams:
            if reqParam not in data:
                data[reqParam] = None
        
        if not isinstance(category, Category) and not isinstance(category, str):
            raise Exception("CATEGORY RAWLOAD ERROR: 'category' must be a Category object or a string representing the category ID.")
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY RAWLOAD ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        catArt = CategoryArtefact(
            category=category,
            artefact=artefact,
            reason=data.get('reason'),
            added=data.get('added')
        )
        
        if withArtefact:
            catArt.getArtefact(includeMetadata=metadataRequired)
        
        return catArt
    
    @staticmethod
    def load(category: Category | str, artefact: Artefact | str, withArtefact: bool=False, metadataRequired: bool=False) -> 'CategoryArtefact | None':
        """Loads a CategoryArtefact instance.

        Args:
            category (Category | str): The category this artefact belongs to (object or ID).
            artefact (Artefact | str): The artefact object or its ID.
            withArtefact (bool, optional): Whether to load the Artefact object when loading. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is required when loading the Artefact. Defaults to False.

        Raises:
            Exception: If the categoryID is not a string.
            Exception: If the artefactID is not a string.

        Returns:
            CategoryArtefact | None: The loaded CategoryArtefact instance.
        """
        
        categoryID = category.id if isinstance(category, Category) else category
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        data = DI.load(CategoryArtefact.ref(categoryID, artefactID))
        if data == None:
            return None
        if isinstance(data, DIError):
            raise Exception("CATEGORYARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("CATEGORYARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return CategoryArtefact.rawLoad(categoryID, artefact, data, withArtefact=withArtefact, metadataRequired=metadataRequired)
    
    @staticmethod
    def ref(categoryID: str, artefactID: str) -> Ref:
        return Ref("categories", categoryID, "members", artefactID)