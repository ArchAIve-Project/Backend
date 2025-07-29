from database import DI, DIRepresentable, DIError
from utils import Ref
from services import Universal
from typing import List, Dict

class AuditLog(DIRepresentable):
    """Represents an audit log entry associated with a user.
    
    Attributes:
        id (str): Unique identifier for the audit log entry. If not provided, it will be auto-generated.
        userID (str): The ID of the user associated with the audit log.
        title (str): The title of the audit log entry.
        description (str): The description of the audit log entry.
        created (str): The timestamp when the audit log entry was created, in ISO format.
        originRef (Ref): The reference to the audit log entry in the database.
    
    Sample usage:
    ```python
    from models import DI, User, AuditLog
    DI.setup()
    
    user = User.load(username="johndoe")
    if not isinstance(user, User):
        raise Exception("Failed to load user - they may not exist. Response: {}".format(user))
    
    log = AuditLog(user=user, title="Login Attempt", description="User logged in successfully.")
    log.save()
    print("Audit log saved:", log)
    ```
    """
    
    def __init__(self, user: 'User | str', title: str, description: str, created: str=None, id: str=None):
        """Initialises an AuditLog instance.

        Args:
            user (User | str): The user associated with the audit log entry.
            title (str): The title of the audit log entry.
            description (str): The description of the audit log entry.
            created (str, optional): The timestamp when the audit log entry was created, in ISO format. Defaults to None.
            id (str, optional): Unique identifier for the audit log entry. If not provided, it will be auto-generated. Defaults to None.

        Raises:
            Exception: If the user is not a User instance or a valid user ID string.
        """
        
        userID = user.id if isinstance(user, User) else user
        if not isinstance(userID, str):
            raise Exception("AUDITLOG INIT ERROR: 'user' must be a User instance or a string representing the user ID.")
        
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        
        self.id = id
        self.userID = userID
        self.title = title
        self.description = description
        self.created = created
        self.originRef = AuditLog.ref(userID, id)
    
    def represent(self):
        return {
            "title": self.title,
            "description": self.description,
            "created": self.created
        }
    
    def save(self):
        return DI.save(self.represent(), self.originRef)
    
    def __str__(self):
        return "AuditLog(id='{}', userID='{}', title='{}', description='{}', created='{}')".format(
            self.id,
            self.userID,
            self.title,
            self.description,
            self.created
        )
    
    @staticmethod
    def rawLoad(userID: str, logID: str, data: dict) -> 'AuditLog':
        reqParams = ['title', 'description', 'created']
        for reqParam in reqParams:
            if reqParam not in data:
                data[reqParam] = None
        
        return AuditLog(
            user=userID,
            title=data.get('title'),
            description=data.get('description'),
            created=data.get('created'),
            id=logID
        )
    
    @staticmethod
    def load(userID: str, logID: str=None) -> 'AuditLog | List[AuditLog] | None':
        """Loads an audit log entry or entries.

        Args:
            userID (str, optional): The ID of the user whose audit logs are to be loaded. Providing only this will load all logs for the user.
            logID (str, optional): The ID of a specific audit log entry to load. Defaults to None.

        Raises:
            Exception: If an error occurs in any part of the loading process.

        Returns:
            AuditLog | List[AuditLog] | None: The loaded audit log entry or entries, or None if not found.
        """
        
        if not isinstance(userID, str):
            raise Exception("AUDITLOG LOAD ERROR: At least a valid `userID` string must be provided.")
        
        if logID != None:
            data = DI.load(AuditLog.ref(userID, logID))
            if isinstance(data, DIError):
                raise Exception("AUDITLOG LOAD ERROR: DIError occurred: {}".format(data))
            if data == None:
                return None
            if not isinstance(data, dict):
                raise Exception("AUDITLOG LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return AuditLog.rawLoad(userID, logID, data)
        else:
            data = DI.load(Ref("auditLogs", userID))
            if isinstance(data, DIError):
                raise Exception("AUDITLOG LOAD ERROR: DIError occurred: {}".format(data))
            if data == None:
                return []
            if not isinstance(data, dict):
                raise Exception("AUDITLOG LOAD ERROR: Failed to load dictionary logs data; response: {}".format(data))
            
            logs: Dict[str, AuditLog] = {}
            for logID in data:
                if isinstance(data[logID], dict):
                    logs[logID] = AuditLog.rawLoad(userID, logID, data[logID])
            
            return list(logs.values())
    
    @staticmethod
    def ref(userID: str, logID: str) -> Ref:
        return Ref("auditLogs", userID, logID)

class User(DIRepresentable):
    """User model representation.
    
    Attributes:
        id (str): Unique identifier for the user.
        username (str): The username of the user.
        email (str): The email of the user.
        pwd (str): The password of the user.
        authToken (str, optional): The auth token of the user. Defaults to None.
        superuser (bool, optional): Whether the user is a superuser. Defaults to False
        lastLogin (str, optional): The last login timestamp of the user. Defaults to None.
        created (str, optional): The creation timestamp of the user. Defaults to None.
        logs (List[AuditLog] | None): A convenience attribute optionally storing the audit logs associated with the user. Populated by `getAuditLogs()`.
    
    Sample usage:
    ```python
    from models import DI, User
    DI.setup()
    
    user = User("johndoe", "john@doe.com", "securepwd")
    user.authToken = "123456"
    user.save()
    
    loadedUser = User.load(username="johndoe", withLogs=True) # load the user and their associated audit logs
    if not isinstance(loadedUser, User):
        raise Exception("Failed to load user - they may not exist. Response: {}".format(loadedUser))
    
    print("User data:")
    print(loadedUser)
    ```
    """
    
    def __init__(self, username: str, email: str, pwd: str, authToken: str=None, superuser: bool=False, lastLogin: str=None, created: str=None, id: str=None):
        """User model representation.

        Args:
            username (str): The username of the user.
            email (str): The email of the user.
            pwd (str): The password of the user.
            authToken (str, optional): The auth token of the user. Defaults to None.
            superuser (bool, optional): Whether the user is a superuser. Defaults to False.
            lastLogin (str, optional): The last login timestamp of the user. Defaults to None.
            created (str, optional): The creation timestamp of the user. Defaults to None. If not provided, it will be set to the current UTC time in ISO format.
            id (str, optional): The unique identifier of the user. Defaults to None. Auto-generated if not provided.
        """
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        
        self.id = id
        self.username = username
        self.email = email
        self.pwd = pwd
        self.authToken = authToken
        self.superuser = superuser
        self.lastLogin = lastLogin
        self.created = created
        self.logs: 'List[AuditLog] | None' = None
        self.originRef = User.ref(id)
    
    def represent(self) -> dict:
        return {
            "username": self.username,
            "email": self.email,
            "pwd": self.pwd,
            "authToken": self.authToken,
            "superuser": self.superuser,
            "lastLogin": self.lastLogin,
            "created": self.created
        }
    
    def __str__(self):
        return """<User instance
ID: {}
Username: {}
Email: {}
Password: {}
Audit Logs:{}
Auth Token: {}
Superuser: {}
Last Login: {}
Created: {} />""".format(
            self.id,
            self.username,
            self.email,
            self.pwd,
            ("\n---\n- " + ("\n- ".join((str(log) if isinstance(log, AuditLog) else "CORRUPTED AUDITLOG AT INDEX '{}'".format(i)) for i, log in enumerate(self.logs))) + "\n---") if isinstance(self.logs, list) and len(self.logs) > 0 else " None or Empty",
            self.authToken,
            self.superuser,
            self.lastLogin,
            self.created
        )

    def save(self, checkSuperuserIntegrity: bool=True):
        """Saves the user to the database.

        Args:
            checkSuperuserIntegrity (bool, optional): Whether to check for superuser integrity. Defaults to True.

        Raises:
            Exception: If an error occurs during saving.
            Exception: If more than one superuser exists.

        Returns:
            bool: Almost exclusively returns `True`.
        """
        
        if self.superuser == True and checkSuperuserIntegrity:
            allUsers = User.load()
            if not isinstance(allUsers, list):
                raise Exception("USER SAVE ERROR: Unexpected User load response format; response: {}".format(allUsers))
            
            for user in allUsers:
                if user.id != self.id and user.superuser == True:
                    raise Exception("USER SAVE ERROR: Cannot save superuser user when another superuser already exists.")
        
        convertedData = self.represent()
        return DI.save(convertedData, self.originRef)
    
    def reload(self, withLogs: bool=False):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("USER RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("USER RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("USER RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id, withLogs).__dict__)
        return True

    def getAuditLogs(self) -> 'List[AuditLog] | None':
        """Retrieves the audit logs associated with the user. Populates `self.logs`.

        Raises:
            Exception: If an error occurs during loading.

        Returns: `List[AuditLog] | None`: A list of audit logs associated with the user, or None if no logs are found.
        """
        
        logs = AuditLog.load(userID=self.id)
        if not (isinstance(logs, list) or logs == None):
            raise Exception("USER GETAUDITLOGS ERROR: Unexpected load response format when retrieving logs for user '{}'; response: {}".format(self.id, logs))
        
        self.logs = sorted(logs, key=lambda log: log.created) if isinstance(logs, list) else logs
        return self.logs

    def newLog(self, title: str, description: str, autoSave: bool=True) -> AuditLog:
        """Creates a new audit log entry for the user.

        Args:
            title (str): The title of the audit log.
            description (str): The description of the audit log.
            autoSave (bool, optional): Whether to automatically save the audit log entry. Defaults to True.

        Returns:
            AuditLog: The new audit log entry.
        """
        
        log = AuditLog(
            user=self.id,
            title=title,
            description=description
        )
        
        if autoSave:
            log.save()
        
        if self.logs is None:
            self.logs = []
        self.logs.append(log)
        
        return log
    
    @staticmethod
    def rawLoad(data: dict, userID: str | None=None, withLogs: bool=False) -> 'User':
        requiredParams = ['username', 'email', 'pwd', 'authToken', 'superuser', 'lastLogin', 'created']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam in []: # add any required params that should be empty dictionaries
                    data[reqParam] = {}
                elif reqParam in []: # add any required params that should be empty lists
                    data[reqParam] = []
                elif reqParam in ['superuser']:
                    data[reqParam] = False
                else:
                    data[reqParam] = None
        
        if not isinstance(data['superuser'], bool):
            data['superuser'] = False
        
        u = User(
            username=data['username'],
            email=data['email'],
            pwd=data['pwd'],
            authToken=data['authToken'],
            superuser=data['superuser'],
            lastLogin=data['lastLogin'],
            created=data['created'],
            id=userID
        )
        
        if withLogs:
            u.getAuditLogs()
        
        return u
    
    @staticmethod
    def load(id: str=None, username: str=None, email: str=None, authToken: str=None, withLogs: bool=False) -> 'User | List[User] | None':
        """Loads a user from the database.

        Args:
            id (str, optional): The ID of the user to load. Defaults to None.
            username (str, optional): The username of the user to load. Defaults to None.
            email (str, optional): The email of the user to load. Defaults to None.
            authToken (str, optional): The auth token of the user to load. Defaults to None.
            withLogs (bool, optional): Optionally load audit logs into a convenient `logs` attribute in the model. Defaults to False. Warning: Might be memory intensive if True.

        Raises:
            Exception: If an error occurs during loading.
            Exception: If the user is not found.
            Exception: If the user data is invalid.
            Exception: If multiple users match the criteria.

        Returns:
            User | List[User] | None: The loaded user(s) or None if not found.
        """
        
        if id != None:
            data = DI.load(User.ref(id))
            if isinstance(data, DIError):
                raise Exception("USER LOAD ERROR: DIError occurred: {}".format(data))
            if data == None:
                return None
            if not isinstance(data, dict):
                raise Exception("USER LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return User.rawLoad(data, id, withLogs)
        else:
            data = DI.load(Ref("users"))
            if data == None:
                return []
            if isinstance(data, DIError):
                raise Exception("USER LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("USER LOAD ERROR: Failed to load dictionary users data; response: {}".format(data))

            users: Dict[str, User] = {}
            for id in data:
                if isinstance(data[id], dict):
                    users[id] = User.rawLoad(data[id], id)

            if username == None and email == None and authToken == None:
                if withLogs:
                    for u in users.values():
                        u.getAuditLogs()
                
                return list(users.values())

            for userID in users:
                targetUser = users[userID]
                if username != None and targetUser.username == username:
                    break
                elif email != None and targetUser.email == email:
                    break
                elif authToken != None and targetUser.authToken == authToken:
                    break

            if targetUser:
                if withLogs:
                    targetUser.getAuditLogs()
                return targetUser
            else:
                return None
    
    @staticmethod
    def getSuperuser() -> 'User | None':
        """ Retrieves the superuser from the database.

        Raises:
            Exception: If more than one superuser exists.
            Exception: If an error occurs during loading.

        Returns:
            User | None: The superuser if found, otherwise None.
        """
        
        allUsers = User.load()
        if not isinstance(allUsers, list):
            raise Exception("USER GETSUPERUSER ERROR: Unexpected User load response format; response: {}".format(allUsers))
        
        su = None
        for user in allUsers:
            if user.superuser == True:
                if su is not None:
                    raise Exception("USER GETSUPERUSER ERROR: More than one superuser found; cannot determine which to return.")
                su = user

        return su

    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("users", id)