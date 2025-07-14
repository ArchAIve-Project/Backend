import os, shutil, json, base64, random, datetime, uuid, functools
from enum import Enum
from typing import List, Dict
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from firebase import FireStorage
from passlib.hash import sha256_crypt as sha
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
load_dotenv()

class Trigger:
    '''
    A class to define triggers for `APScheduler` based jobs.
    
    Implementation variations:
    ```python
    from apscheduler.triggers.date import DateTrigger
    import datetime
    from services import Trigger
    
    ## Immediate trigger
    immediateTrigger = Trigger()
    
    ## Interval trigger
    intervalTrigger = Trigger(type='interval', seconds=5, minutes=3, hours=2) # Triggers every 2 hours 3 minutes and 5 seconds (cumulatively)
    
    ## Date trigger
    dateTrigger = Trigger(type='date', triggerDate=(datetime.datetime.now() + datetime.timedelta(seconds=5))) # Triggers 5 seconds later
    
    ## Custom trigger
    customTrigger = Trigger(customAPTrigger=DateTrigger(run_date=(datetime.datetime.now() + datetime.timedelta(seconds=5))) # Custom APScheduler trigger
    ```
    '''
    
    def __init__(self, type='interval', seconds=None, minutes=None, hours=None, triggerDate: datetime.datetime=None, customAPTrigger: BaseTrigger=None) -> None:
        self.type = type
        self.immediate = seconds == None and minutes == None and hours == None and triggerDate == None
        self.seconds = seconds if seconds != None else 0
        self.minutes = minutes if minutes != None else 0
        self.hours = hours if hours != None else 0
        self.triggerDate = triggerDate
        self.customAPTrigger = customAPTrigger

class AsyncProcessor:
    """
    Async processor uses the `apscheduler` library to run functions asynchronously. Specific intervals can be set if needed as well.
    
    Manipulate the underling scheduler with the `scheduler` attribute.
    
    Sample Usage:
    ```python
    asyncProcessor = AsyncProcessor()
    
    def greet():
        print("Hi!")
    
    jobID = asyncProcessor.addJob(greet, trigger=Trigger(type='interval', seconds=5)) # This is all you need to add the job
    
    ## Pausing the scheduler
    asyncProcessor.pause()
    
    ## Resuming the scheduler
    asyncProcessor.resume()
    
    ## Shutting down the scheduler
    asyncProcessor.shutdown()
    
    ## To stop a job, use the jobID
    asyncProcessor.scheduler.remove_job(jobID) ## .pause_job() and .resume_job() can also be used to manipulate the job itself
    
    ## Disable logging with Logger
    asyncProcessor.logging = False
    ```
    
    For trigger options, see the `Trigger` class.
    """
    
    def __init__(self, paused=False, logging=True) -> None:
        self.id = uuid.uuid4().hex
        self.scheduler = BackgroundScheduler()
        self.scheduler.start(paused=paused)
        self.logging = logging
        
        self.log("Scheduler initialised in {} mode.".format("paused" if paused else "active"))
        
    def log(self, message):
        if self.logging == True:
            Logger.log("ASYNCPROCESSOR: {}: {}".format(self.id, message))
    
    def shutdown(self):
        self.scheduler.shutdown()
        self.log("Scheduler shutdown.")
    
    def pause(self):
        self.scheduler.pause()
        self.log("Scheduler paused.")
    
    def resume(self):
        self.scheduler.resume()
        self.log("Scheduler resumed.")
    
    def addJob(self, function, *args, trigger: Trigger=None, **kwargs):
        """
        Adds a job to the scheduler.
        """
        if trigger == None: trigger = Trigger()
        
        job = None
        if trigger.customAPTrigger != None:
            job = self.scheduler.add_job(function, trigger.customAPTrigger, args=args, kwargs=kwargs)
            self.log("Job for '{}' added with custom trigger.".format(function.__name__))
        elif trigger.immediate:
            job = self.scheduler.add_job(function, args=args, kwargs=kwargs)
            self.log("Job for '{}' added with immediate trigger.".format(function.__name__))
        elif trigger.type == "date":
            job = self.scheduler.add_job(function, DateTrigger(run_date=trigger.triggerDate), args=args, kwargs=kwargs)
            self.log("Job for '{}' added with trigger date: {}.".format(function.__name__, trigger.triggerDate.isoformat()))
        else:
            job = self.scheduler.add_job(function, 'interval', seconds=trigger.seconds, minutes=trigger.minutes, hours=trigger.hours, args=args, kwargs=kwargs)
            self.log("Job for '{}' added with trigger: {} seconds, {} minutes, {} hours.".format(function.__name__, trigger.seconds, trigger.minutes, trigger.hours))
            
        return job.id

class ThreadManager:
    '''Manage multiple `AsyncProcessor`s with this class.
    
    Spin up multiple processors and add jobs to them. Each processor spins up one or more separate threads and schedules jobs to run in the background.
    
    **Recommended Usage:**
    ```python
    import time
    from services import ThreadManager, AsyncProcessor, Trigger
    
    def sample_work(some_param):
        print(f"Working with {some_param}...")
        time.sleep(2)
        print(f"Done working with {some_param}!")
    
    ThreadManager.initDefault() # Initializes a default processor at `ThreadManager.defaultProcessor`
    id = ThreadManager.defaultProcessor.addJob(sample_work, "example_param", trigger=Trigger(type='interval', seconds=10)) # Adds a job to the default processor
    
    print(f"Job ID: {id}")
    ```
    
    **Advanced Usage with Multiple Processors (Caution):**
    ```python
    import time
    from services import ThreadManager, AsyncProcessor, Trigger
    
    # Spinning up a custom processor
    processor = ThreadManager.new(name="newScheduler", source="example.py") # Returns an `AsyncProcessor` instance. `name` must be unique.
    processor.addJob(lambda: print("Hello from the new thread!"), trigger=Trigger(type='interval', seconds=5))
    
    print("Current processors:", ThreadManager.list())
    print("Processor info:", ThreadManager.info("newThread"))

    time.sleep(10)
    
    processor2 = ThreadManager.getProcessorWithName("newScheduler") # Get the processor for the `newScheduler`
    processor2.addJob(lambda: print("This is another job in the same thread!"), trigger=Trigger(type='interval', seconds=10))
    
    time.sleep(25)

    ThreadManager.closeThread("newScheduler") # Close the processor
    # ThreadManager.shutdown() # Shutdown all processors
    ```
    
    Methods:
    - `list()`: Returns a list of all processor names.
    - `initDefault()`: Initializes a default processor named "default" and returns it.
    - `new(name: str, source: str, paused: bool=False, logging: bool=True)`: Creates a new processor with the given name and source. Returns an `AsyncProcessor` instance or an error message if the name is not unique.
    - `info(name: str)`: Returns information about the processor with the given name, including the processor, name, source, and creation time.
    - `getProcessorWithName(name: str)`: Returns the `AsyncProcessor` instance for the processor with the given name, or `None` if it does not exist.
    - `getProcessorWithID(id: str)`: Returns the `AsyncProcessor` instance for the processor with the given ID, or `None` if it does not exist.
    - `closeThread(name: str)`: Closes the processor with the given name and returns `True` if successful, or `False` if the processor does not exist.
    - `shutdown()`: Shuts down all threads and returns `True` if successful.
    
    Note: Be careful when using this class. It is very powerful and creates multiple real background threads. Close threads properly when no longer needed.
    '''
    
    data = {}
    defaultProcessor: AsyncProcessor | None = None
    
    @staticmethod
    def list() -> list[str]:
        '''Returns a list of all thread names managed by the ThreadManager.'''
        return list(ThreadManager.data.keys())
    
    @staticmethod
    def initDefault() -> AsyncProcessor:
        if "default" in ThreadManager.data:
            raise Exception("ERROR: Default thread already exists.")

        ThreadManager.defaultProcessor = ThreadManager.new(name="default", source="main.py")
        return ThreadManager.defaultProcessor
    
    @staticmethod
    def new(name: str, source: str, paused: bool=False, logging: bool=True) -> AsyncProcessor | str:
        '''Creates a new thread with the given name and source. Returns an `AsyncProcessor` instance or an error message if the name is not unique.'''
        if name in ThreadManager.data:
            return "ERROR: A thread with that name already exists. Name must be unique."
        
        processor = AsyncProcessor(paused=paused, logging=logging)
        ThreadManager.data[name] = {
            "processor": processor,
            "name": name,
            "source": source,
            "created": Universal.utcNowString()
        }
        
        Logger.log("THREADMANAGER NEW: New thread with name '{}' created.".format(name))
        
        return processor

    @staticmethod
    def info(name: str) -> dict | None:
        '''Returns information about the thread with the given name, including the processor, name, source, and creation time. Returns `None` if the thread does not exist.'''
        return ThreadManager.data.get(name)
    
    @staticmethod
    def getProcessorWithName(name: str) -> AsyncProcessor | None:
        '''Returns the `AsyncProcessor` instance for the thread with the given name, or `None` if it does not exist.'''
        if name in ThreadManager.data:
            return ThreadManager.data[name]["processor"]
        else:
            return None
    
    @staticmethod
    def getProcessorWithID(id: str) -> AsyncProcessor | None:
        '''Returns the `AsyncProcessor` instance for the thread with the given ID, or `None` if it does not exist.'''
        for threadName in ThreadManager.list():
            if ThreadManager.data[threadName]["processor"].id == id:
                return ThreadManager.data[threadName]["processor"]
        return None
    
    @staticmethod
    def closeThread(name: str) -> bool:
        '''Closes the thread with the given name and returns `True` if successful, or `False` if the thread does not exist.'''
        processor = ThreadManager.getProcessorWithName(name)
        if processor == None:
            return False

        processor.shutdown()
        del ThreadManager.data[name]
        
        if name == "default":
            ThreadManager.defaultProcessor = None
        
        Logger.log("THREADMANAGER CLOSETHREAD: Thread '{}' closed successfully.".format(name))
        return True
    
    @staticmethod
    def shutdown() -> bool:
        '''Shuts down all threads and returns `True` if successful.'''
        for thread in ThreadManager.list():
            ThreadManager.closeThread(thread)
        
        ThreadManager.defaultProcessor = None
        
        Logger.log("THREADMANAGER SHUTDOWN: Shutdown complete.")
        return True

class Encryption:
    @staticmethod
    def encodeToB64(inputString):
        '''Encodes a string to base64'''
        hash_bytes = inputString.encode("ascii")
        b64_bytes = base64.b64encode(hash_bytes)
        b64_string = b64_bytes.decode("ascii")
        return b64_string
    
    @staticmethod
    def decodeFromB64(encodedHash):
        '''Decodes a base64 string to a string'''
        b64_bytes = encodedHash.encode("ascii")
        hash_bytes = base64.b64decode(b64_bytes)
        hash_string = hash_bytes.decode("ascii")
        return hash_string
  
    @staticmethod
    def isBase64(encodedHash):
        '''Checks if a string is base64'''
        try:
            hashBytes = encodedHash.encode("ascii")
            return base64.b64encode(base64.b64decode(hashBytes)) == hashBytes
        except Exception:
            return False

    @staticmethod
    def encodeToSHA256(string):
        '''Encodes a string to SHA256'''
        return sha.hash(string)
  
    @staticmethod
    def verifySHA256(inputString, hash):
        '''Verifies a string against a SHA256 hash using the `sha` module directly'''
        return sha.verify(inputString, hash)
  
    @staticmethod
    def convertBase64ToSHA(base64Hash):
        '''Converts a base64 string to a SHA256 hash'''
        return Encryption.encodeToSHA256(Encryption.decodeFromB64(base64Hash))
    
class Universal:
    '''This class contains universal methods and variables that can be used across the entire project. Project-wide standards and conventions (such as datetime format) are also defined here.'''

    systemWideStringDatetimeFormat = "%Y-%m-%d %H:%M:%S"
    copyright = "© 2025 The ArchAIve Team. All Rights Reserved."
    version = None
    store = {}
    
    @staticmethod
    def getVersion():
        if Universal.version != None:
            return Universal.version
        if not os.path.isfile(os.path.join(os.getcwd(), 'version.txt')):
            return None
        else:
            with open('version.txt', 'r') as f:
                fileData = f.read()
                Universal.version = fileData
                return fileData

    @staticmethod
    def generateUniqueID(customLength=None, notIn=[]):
        if customLength == None:
            return uuid.uuid4().hex
        else:
            source = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            out = ''.join(random.choices(source, k=customLength))
            while out in notIn:
                out = ''.join(random.choices(source, k=customLength))
            return out
    
    @staticmethod
    def utcNow():
        return datetime.datetime.now(datetime.timezone.utc)
    
    @staticmethod
    def utcNowString():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    @staticmethod
    def fromUTC(utcString):
        return datetime.datetime.fromisoformat(utcString)
        
class Logger:
    file = "logs.txt"
    
    '''## Intro
    A class offering silent and quick logging services.

    Explicit permission must be granted by setting `LOGGING_ENABLED` to `True` in the `.env` file. Otherwise, all logging services will be disabled.
    
    ## Usage:
    ```
    Logger.setup() ## Optional

    Logger.log("Hello world!") ## Adds a log entry to the logs.txt database file, if permission was granted.
    ```

    ## Advanced:
    Activate Logger's management console by running `Logger.manageLogs()`. This will allow you to read and destroy logs in an interactive manner.
    '''
    
    @staticmethod
    def checkPermission():
        return "LOGGING_ENABLED" in os.environ and os.environ["LOGGING_ENABLED"] == 'True'

    @staticmethod
    def setup():
        if Logger.checkPermission():
            try:
                if not os.path.exists(os.path.join(os.getcwd(), Logger.file)):
                    with open(Logger.file, "w") as f:
                        f.write("{}UTC {}\n".format(Universal.utcNowString(), "LOGGER: Logger database file setup complete."))
            except Exception as e:
                print("LOGGER SETUP ERROR: Failed to setup logs.txt database file. Setup permissions have been granted. Error: {}".format(e))

        return

    @staticmethod
    def log(message, debugPrintExplicitDeny=False):
        if "DEBUG_MODE" in os.environ and os.environ["DEBUG_MODE"] == 'True' and (not debugPrintExplicitDeny):
            print("LOG: {}".format(message))
        if Logger.checkPermission():
            try:
                with open(Logger.file, "a") as f:
                    f.write("{}UTC {}\n".format(Universal.utcNowString(), message))
            except Exception as e:
                print("LOGGER LOG ERROR: Failed to log message. Error: {}".format(e))
        
        return
    
    @staticmethod
    def destroyAll():
        try:
            if os.path.exists(os.path.join(os.getcwd(), Logger.file)):
                os.remove(Logger.file)
        except Exception as e:
            print("LOGGER DESTROYALL ERROR: Failed to destroy logs.txt database file. Error: {}".format(e))

    @staticmethod
    def readAll(explicitLogsFile=None):
        if not Logger.checkPermission():
            return "ERROR: Logging-related services do not have permission to operate."
        logsFile = Logger.file if explicitLogsFile == None else explicitLogsFile
        try:
            if os.path.exists(os.path.join(os.getcwd(), logsFile)):
                with open(logsFile, "r") as f:
                    logs = f.readlines()
                    for logIndex in range(len(logs)):
                        logs[logIndex] = logs[logIndex].replace("\n", "")
                    return logs
            else:
                return []
        except Exception as e:
            print("LOGGER READALL ERROR: Failed to check and read logs file. Error: {}".format(e))
            return "ERROR: Failed to check and read logs file. Error: {}".format(e)
      
    @staticmethod
    def manageLogs(explicitLogsFile=None):
        permission = Logger.checkPermission()
        if not permission:
            print("LOGGER: Logging-related services do not have permission to operate. Set LoggingEnabled to True in .env file to enable logging.")
            return
    
        print("LOGGER: Welcome to the Logging Management Console.")
        while True:
            print("""
Commands:
    read <number of lines, e.g 50 (optional)>: Reads the last <number of lines> of logs. If no number is specified, all logs will be displayed.
    destroy: Destroys all logs.
    exit: Exit the Logging Management Console.
""")
    
            userChoice = input("Enter command: ")
            userChoice = userChoice.lower()
            while not userChoice.startswith("read") and (userChoice != "destroy") and (userChoice != "exit"):
                userChoice = input("Invalid command. Enter command: ")
                userChoice = userChoice.lower()
    
            if userChoice.startswith("read"):
                allLogs = Logger.readAll(explicitLogsFile=explicitLogsFile)
                targetLogs = []

                userChoice = userChoice.split(" ")

                # Log filtering feature
                if len(userChoice) == 1:
                    targetLogs = allLogs
                elif userChoice[1] == ".filter":
                    if len(userChoice) < 3:
                        print("Invalid log filter. Format: read .filter <keywords>")
                        continue
                    else:
                        try:
                            keywords = userChoice[2:]
                            for log in allLogs:
                                logTags = log[36::]
                                logTags = logTags[:logTags.find(":")].upper().split(" ")

                                ## Check if log contains all keywords
                                containsAllKeywords = True
                                for keyword in keywords:
                                    if keyword.upper() not in logTags:
                                        containsAllKeywords = False
                                        break
                                
                                if containsAllKeywords:
                                    targetLogs.append(log)
                                
                            print("Filtered logs with keywords: {}".format(keywords))
                            print()
                        except Exception as e:
                            print("LOGGER: Failed to parse and filter logs. Error: {}".format(e))
                            continue
                else:
                    logCount = 0
                    try:
                        logCount = int(userChoice[1])
                        if logCount > len(allLogs):
                            logCount = len(allLogs)
                        elif logCount <= 0:
                            raise Exception("Invalid log count. Must be a positive integer above 0 lower than or equal to the total number of logs.")
                        
                        targetLogs = allLogs[-logCount::]
                    except Exception as e:
                        print("LOGGER: Failed to read logs. Error: {}".format(e))
                        continue

                logCount = len(targetLogs)
                print()
                print("Displaying {} log entries:".format(logCount))
                print()
                for log in targetLogs:
                    print("\t{}".format(log))
            elif userChoice == "destroy":
                Logger.destroyAll()
                print("LOGGER: All logs destroyed.")
            elif userChoice == "exit":
                print("LOGGER: Exiting Logging Management Console...")
                break
    
        return

class FileOps:
    @staticmethod
    def exists(path: str, type: str="folder"):
        '''Check the existence of a file/folder. Provide 'path' and 'type'.
        Valid values for 'type' include 'folder' and 'file'.
        Raises `ValueError` if an invalid type value is provided.'''
        if type not in ["folder", "file"]:
            raise ValueError("Invalid type provided. Must be 'folder' or 'file'.")
        
        if type == "file":
            return os.path.isfile(path)
        else:
            return os.path.isdir(path)
    
    @staticmethod
    def createFolder(relativePath: str) -> bool | str:
        if FileOps.exists(os.path.join(os.getcwd(), relativePath), type="folder"):
            return True
        
        try:
            os.mkdir(os.path.join(os.getcwd(), relativePath))
            return True
        except Exception as e:
            return "ERROR: Failed to create folder '{}'; error: {}".format(relativePath, e)
    
    @staticmethod
    def getFilenames(relativeFolderPath: str) -> list[str] | str:
        if not FileOps.exists(os.path.join(os.getcwd(), relativeFolderPath), type="folder"):
            return []
        
        try:
            files = os.listdir(os.path.join(os.getcwd(), relativeFolderPath))
            return files
        except Exception as e:
            return "ERROR: Failed to list files in folder '{}'; error: {}".format(relativeFolderPath, e)
    
    @staticmethod
    def deleteFolder(relativePath: str) -> bool | str:
        if not FileOps.exists(os.path.join(os.getcwd(), relativePath), type="folder"):
            return True
        
        try:
            shutil.rmtree(os.path.join(os.getcwd(), relativePath))
            return True
        except Exception as e:
            return "ERROR: Failed to delete folder '{}'; error: {}".format(relativePath, e)
    
    @staticmethod
    def deleteFile(relativePath: str) -> bool | str:
        if not FileOps.exists(os.path.join(os.getcwd(), relativePath), type="file"):
            return True
        
        try:
            os.remove(os.path.join(os.getcwd(), relativePath))
            return True
        except Exception as e:
            return "ERROR: Failed to remove file '{}'; error: {}".format(relativePath, e)
    
    @staticmethod
    def getFileExtension(path: str):
        return os.path.splitext(path)[1].replace(".", "")