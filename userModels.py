from database import DI, DIRepresentable, DIError
from utils import Ref
from services import Universal
from typing import List, Dict

class AuditLog(DIRepresentable):
    def __init__(self, user: 'User | str', title: str, description: str, created: str=None, id: str=None):
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
    def load(userID: str=None, logID: str=None) -> 'AuditLog | List[AuditLog] | None':
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
    
    Sample usage:
    ```python
    from models import DI, User
    DI.setup()
    
    user = User("johndoe", "john@doe.com", "securepwd")
    user.authToken = "123456"
    user.save()
    
    loadedUser = User.load(username="johndoe")
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
            ("\n---\n- " + ("\n- ".join((str(log) if isinstance(log, AuditLog) else "CORRUPTED AUDITLOG AT INDEX '{}'".format(i)) for i, log in enumerate(self.logs))) + "\n---") if isinstance(self.logs, list) else " None",
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
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("USER RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("USER RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("USER RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    @staticmethod
    def rawLoad(data: dict, userID: str | None=None) -> 'User':
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
        
        return User(
            username=data['username'],
            email=data['email'],
            pwd=data['pwd'],
            authToken=data['authToken'],
            superuser=data['superuser'],
            lastLogin=data['lastLogin'],
            created=data['created'],
            id=userID
        )
    
    @staticmethod
    def load(id: str=None, username: str=None, email: str=None, authToken: str=None) -> 'User | List[User] | None':
        """Loads a user from the database.

        Args:
            id (str, optional): The ID of the user to load. Defaults to None.
            username (str, optional): The username of the user to load. Defaults to None.
            email (str, optional): The email of the user to load. Defaults to None.
            authToken (str, optional): The auth token of the user to load. Defaults to None.

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
            
            return User.rawLoad(data, id)
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
                return list(users.values())

            for userID in users:
                targetUser = users[userID]
                if username != None and targetUser.username == username:
                    return targetUser
                elif email != None and targetUser.email == email:
                    return targetUser
                elif authToken != None and targetUser.authToken == authToken:
                    return targetUser

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