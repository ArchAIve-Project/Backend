import os, datetime, copy
from typing import List
from firebase_admin import db, storage, credentials, initialize_app
from google.cloud.storage.blob import Blob
from google.api_core.page_iterator import HTTPIterator
from services import LiteStore, Logger
from dotenv import load_dotenv
load_dotenv()

class FireConn:
    '''A class that manages the admin connection to Firebase via the firebase_admin module.
    
    Explicit permission has to be granted by setting `FireConnEnabled` to `True` in the .env file. `serviceAccountKey.json` file must be in working directory to provide credentials for the connection. Obtain one on the Firebase Console (under Service Accounts > Firebase Admin SDK > Generate new private key).

    Usage:

    ```
    response = FireConn.connect()
    if response != True:
        print("Error in setting up FireConn; error: " + response)
        sys.exit(1)
    ```

    NOTE: This class is not meant to be instantiated. Other services relying on connection via firebase_admin module need to run the `connect` method in this class first. If permission is not granted, dependant services may not be able to operate.
    '''
    connected = False

    @staticmethod
    def checkPermissions():
        return ("FireConnEnabled" in os.environ and os.environ["FireConnEnabled"] == "True")

    @staticmethod
    def connect():
        '''Returns True upon successful connection.'''
        if not FireConn.checkPermissions():
            return "ERROR: Firebase connection permissions are not granted."
        if not os.path.exists("serviceAccountKey.json"):
            return "ERROR: Failed to connect to Firebase. The file serviceAccountKey.json was not found."
        else:
            if 'RTDB_URL' not in os.environ:
                return "ERROR: Failed to connect to Firebase. RTDB_URL environment variable not set in .env file."
            if 'STORAGE_URL' not in os.environ:
                return "ERROR: Failed to connect to Firebase. STORAGE_URL environment variable not set in .env file."
            try:
                ## Firebase
                cred_obj = credentials.Certificate(os.path.join(os.getcwd(), "serviceAccountKey.json"))
                default_app = initialize_app(cred_obj, {
                    'databaseURL': os.environ["RTDB_URL"],
                    'storageBucket': os.environ["STORAGE_URL"],
                    'httpTimeout': 10
                })
                FireConn.connected = True
            except Exception as e:
                return "ERROR: Error occurred in connecting to RTDB; error: {}".format(e)
            return True
        
class FireRTDB:
    '''A class to update Firebase Realtime Database (RTDB) references with data.

    Explicit permission has to be granted by setting `FireRTDBEnabled` to `True` in the .env file.

    Usage:
    ```
    if FireRTDB.checkPermissions():
        data = {"name": "John Appleseed"}
        FireRTDB.setRef(data, refPath="/")
        fetchedData = FireRTDB.getRef(refPath="/")
        print(fetchedData) ## same as data defined above
    ```

    Advanced Usage:
    ```
    ## DB translation
    db = {"jobs": {}}
    safeForCloudDB = FireRTDB.translateForCloud(db) ## {"jobs": 0}
    # safeForLocalDB = FireRTDB.translateForLocal(safeForCloudDB) ## {"jobs": {}}
    ```

    `FireRTDB.translateForCloud` and `FireRTDB.translateForLocal` are used to translate data structures for cloud and local storage respectively. This is because Firebase does not allow null objects to be stored in the RTDB. This method converts null objects to a value that can be stored in the RTDB. The following conversions are performed:
    - Converts `{}` to `0` and `[]` to `1` (for cloud storage)
    - Converts `0` to `{}` and `1` to `[]` (for local storage)

    NOTE: This class is not meant to be instantiated. `FireConn.connect()` method must be executed before executing any other methods in this class.
    '''

    @staticmethod
    def checkPermissions():
        '''Returns True if permission is granted, otherwise returns False.'''
        if 'FireRTDBEnabled' in os.environ and os.environ['FireRTDBEnabled'] == 'True':
            return True
        return False

    @staticmethod
    def clearRef(refPath="/"):
        '''Returns True upon successful update. Providing `refPath` is optional; will be the root reference if not provided.'''
        if not FireRTDB.checkPermissions():
            return "ERROR: FireRTDB service operation permission denied."
        try:
            ref = db.reference(refPath)
            ref.delete()
        except Exception as e:
            return "ERROR: Error occurred in clearing children at that ref; error: {}".format(e)
        return True

    @staticmethod
    def setRef(data, refPath="/"):
        '''Returns True upon successful update. Providing `refPath` is optional; will be the root reference if not provided.'''
        if not FireRTDB.checkPermissions():
            raise Exception("ERROR: FireRTDB service operation permission denied.")
        try:
            ref = db.reference(refPath)
            ref.set(data)
        except Exception as e:
            raise Exception("ERROR: Error occurred in setting data at that ref; error: {}".format(e))
        return True

    @staticmethod
    def getRef(refPath="/"):
        '''Returns a dictionary of the data at the specified ref. Providing `refPath` is optional; will be the root reference if not provided.'''
        if not FireRTDB.checkPermissions():
            raise Exception("ERROR: FireRTDB service operation permission denied.")
        data = None
        try:
            ref = db.reference(refPath)
            data = ref.get()
        except Exception as e:
            raise Exception("ERROR: Error occurred in getting data from that ref; error: {}".format(e))
        
        return data
        
    @staticmethod
    def recursiveReplacement(obj, purpose):
        dictValue = {} if purpose == 'cloud' else 0
        dictReplacementValue = 0 if purpose == 'cloud' else {}

        arrayValue = [] if purpose == 'cloud' else 1
        arrayReplacementValue = 1 if purpose == 'cloud' else []

        data = copy.deepcopy(obj)

        for key in data:
            if isinstance(data, list):
                # This if statement forbids the following sub-data-structure: [{}, 1, {}] (this is an example)
                continue

            if isinstance(data[key], dict):
                if data[key] == dictValue:
                    data[key] = dictReplacementValue
                else:
                    data[key] = FireRTDB.recursiveReplacement(data[key], purpose)
            elif isinstance(data[key], list):
                if data[key] == arrayValue:
                    data[key] = arrayReplacementValue
                else:
                    data[key] = FireRTDB.recursiveReplacement(data[key], purpose)
            elif isinstance(data[key], bool):
                continue
            elif isinstance(data[key], int) and purpose == 'local':
                if data[key] == 0:
                    data[key] = {}
                elif data[key] == 1:
                    data[key] = []

        return data
    
    @staticmethod
    def translateForLocal(fetchedData, rootTranslatable=False):
        '''Returns a translated data structure that can be stored locally.'''
        if fetchedData == None:
            return None
        if rootTranslatable:
            if fetchedData == 0:
                return {}
            elif fetchedData == 1:
                return []
            elif not (isinstance(fetchedData, dict) or isinstance(fetchedData, list)):
                return fetchedData
        
        tempData = copy.deepcopy(fetchedData)

        try:
            # Null object replacement
            tempData = FireRTDB.recursiveReplacement(obj=tempData, purpose='local')

            # Further translation here
            
        except Exception as e:
            return "ERROR: Error in translating fetched RTDB data for local system use; error: {}".format(e)
        
        return tempData
    
    @staticmethod
    def translateForCloud(loadedData, rootTranslatable=False):
        '''Returns a translated data structure that can be stored in the cloud.'''
        if loadedData == None:
            return None
        if rootTranslatable:
            if loadedData == {}:
                return 0
            elif loadedData == []:
                return 1
            elif not (isinstance(loadedData, dict) or isinstance(loadedData, list)):
                return loadedData

        tempData = copy.deepcopy(loadedData)

        # Further translation here

        # Null object replacement
        tempData = FireRTDB.recursiveReplacement(obj=tempData, purpose='cloud')

        return tempData

class FireStorage:
    '''A class to upload and download files to and from Firebase Storage.
    
    Explicit permission has to be granted by setting `FireStorageEnabled` to `True` in the .env file. `STORAGE_URL` variable must be set in .env file. Obtain one by copying the bucket URL in Storage on the Firebase Console.

    Usage:
    ```
    response = FireStorage.uploadFile(localFilePath="test.txt", filename="test.txt")
    if response != True:
        print("Error in uploading file; error: " + response)
        sys.exit(1)

    response = FireStorage.downloadFile(localFilePath="downloadedTest.txt", filename="test.txt")
    if response != True:
        print("Error in downloading file; error: " + response)
        sys.exit(1)
    ```
    
    Methods:
    - `checkPermissions()`: Returns True if permission is granted, otherwise returns False.
    - `listFiles(namesOnly: bool=False)`: Returns a list of filenames in the Firebase Storage bucket. If `namesOnly` is True, returns a list of filenames only. (`List[str]`). If not, returns a `google.api_core.page_iterator.HTTPIterator` of `Blob` objects.
    - `clearStorage()`: Deletes all files in the Firebase Storage bucket. Returns True upon successful deletion.
    - `getFileInfo(filename: str, blob: Blob=None, metadataOnly=True)`: Returns a metadata dictionary of a file in Firebase Storage. Set `metadataOnly` to False to obtain the `Blob` object instead. Provide a blob directly to avoid fetching it again from the storage. Returns `None` if the file does not exist, or a string prefixed with 'ERROR: ' if an error occurred in the process.
    - `getFileSignedURL(filename: str, expiration: datetime.timedelta=None, allowCache: bool=True, updateCache: bool=True)`: Returns the signed URL of a file in Firebase Storage. If `allowCache` is True, caches the URL for 1 hour by default. If `updateCache` is True, updates the cache with the new URL.
    - `getFilePublicURL(filename: str)`: Returns the public URL of a file in Firebase Storage.
    - `changeFileACL(filename: str, private: bool=True)`: Changes the ACL of a file in Firebase Storage to private or public. Returns True upon successful change.
    - `uploadFile(localFilePath, filename=None)`: Uploads a file to Firebase Storage. Returns True upon successful upload.
    - `downloadFile(localFilePath, filename=None)`: Downloads a file from Firebase Storage. Returns True upon successful download.
    - `deleteFile(filename)`: Deletes a file from Firebase Storage. Returns True upon successful deletion.
    - `deleteFiles(filenames: List[str])`: Deletes multiple files from Firebase Storage. Returns True upon successful deletion.

    NOTE: This class is not meant to be instantiated. `FireConn.connect()` method must be executed before executing any other methods in this class.
    '''
    
    # signedURLCache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def checkPermissions():
        '''Returns True if permission is granted, otherwise returns False.'''
        return ('FireStorageEnabled' in os.environ and os.environ['FireStorageEnabled'] == 'True')
    
    @staticmethod
    def listFiles(namesOnly: bool=False, ignoreFolders: bool=False) -> List[Blob] | List[str]:
        '''Returns a list of filenames in the Firebase Storage bucket.
        
        If `namesOnly` is True, returns a list of filenames only. (`List[str]`)
        If not, though the type hint is `List[Blob]`, a `google.api_core.page_iterator.HTTPIterator` of `Blob` objects is returned, which can be iterated over to get the `Blob` objects.
        '''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            bucket = storage.bucket()
            blobs: List[Blob] = bucket.list_blobs()
            
            if ignoreFolders:
                blobs = [blob for blob in blobs if not blob.name.endswith("/")]
            
            if namesOnly:
                return [blob.name for blob in blobs]
            else:
                return blobs
        except Exception as e:
            return "ERROR: Error occurred in listing files in cloud storage; error: {}".format(e)
    
    @staticmethod
    def clearStorage():
        '''Deletes all files in the Firebase Storage bucket. Returns True upon successful deletion.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            bucket = storage.bucket()
            blobs: List[Blob] = bucket.list_blobs()
            for blob in blobs:
                blob.delete()
        except Exception as e:
            return "ERROR: Error occurred in clearing cloud storage; error: {}".format(e)
        return True
    
    @staticmethod
    def getFileInfo(filename: str, blob: Blob=None, metadataOnly=True) -> dict | Blob | None | str:
        '''Returns a metadata dictionary of a file in Firebase Storage. Set `metadataOnly` to False to obtain the `Blob` object instead.
        
        `None` will be returned if the file does not exist. A string prefixed with 'ERROR: ' will be returned if an error occurred in the process.
        '''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            if blob is None:
                bucket = storage.bucket()
                blob = bucket.get_blob(filename)
                if blob is None:
                    return None
            
            if metadataOnly:
                return {
                    "id": blob.id,
                    "name": blob.name,
                    "path": blob.path,
                    "owner": blob.owner,
                    "publicURL": blob.public_url,
                    "contentType": blob.content_type,
                    "contentDisposition": blob.content_disposition,
                    "contentEncoding": blob.content_encoding,
                    "created": blob.time_created,
                    "updated": blob.updated,
                    "size": blob.size,
                    "metadata": blob.metadata
                }
            else:
                return blob
        except Exception as e:
            return "ERROR: Error occurred in getting file info from cloud storage; error: {}".format(e)
    
    @staticmethod
    def getFileSignedURL(filename: str, expiration: datetime.timedelta=None, allowCache: bool=True, updateCache: bool=True) -> str:
        '''Returns the signed URL of a file in Firebase Storage.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        
        if expiration is None:
            expiration = datetime.timedelta(hours=1)
        
        cacheData: dict | None = None
        if allowCache:
            cacheData = LiteStore.read("signedURLCache") or {}
            urlData = cacheData.get(filename, None)
            if urlData is not None:
                if datetime.datetime.fromisoformat(urlData['expiration']) > datetime.datetime.now(datetime.timezone.utc):
                    return urlData['url']
                else:
                    del cacheData[filename]
                    res = LiteStore.set("signedURLCache", cacheData)
                    if res != True:
                        Logger.log("FIRESTORAGE GETFILESIGNEDURL WARNING: Failed to update signedURLCache; response: {}".format(res))
        
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            
            expirationDatetime = datetime.datetime.now(tz=datetime.timezone.utc) + expiration
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=expirationDatetime,
                method="GET"
            )
            
            if updateCache:
                if cacheData is None:
                    cacheData = LiteStore.read("signedURLCache") or {}

                cacheData[filename] = {
                    "url": url,
                    "expiration": expirationDatetime.isoformat()
                }
                
                res = LiteStore.set("signedURLCache", cacheData)
                if res != True:
                    Logger.log("FIRESTORAGE GETFILESIGNEDURL WARNING: Failed to update signedURLCache; response: {}".format(res))
            return url
        except Exception as e:
            return "ERROR: Error in generating signed URL for file in cloud storage; error: {}".format(e)
    
    @staticmethod
    def getFilePublicURL(filename: str) -> str:
        '''Returns the public URL of a file in Firebase Storage.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            return blob.public_url
        except Exception as e:
            return "ERROR: Error occurred in getting file public URL from cloud storage; error: {}".format(e)
    
    @staticmethod
    def changeFileACL(filename: str, private: bool=True):
        '''Changes the ACL of a file in Firebase Storage to private or public. Returns True upon successful change.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            if private:
                blob.make_private()
            else:
                blob.make_public()
        except Exception as e:
            return "ERROR: Error occurred in changing file ACL in cloud storage; error: {}".format(e)
        return True

    @staticmethod
    def uploadFile(localFilePath, filename=None):
        '''Returns True upon successful upload.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        if filename == None:
            filename = os.path.basename(localFilePath)
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            blob.upload_from_filename(localFilePath)
        except Exception as e:
            return "ERROR: Error occurred in uploading file to cloud storage; error: {}".format(e)
        return True

    @staticmethod
    def downloadFile(localFilePath, filename=None):
        '''Returns True upon successful download.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        if filename == None:
            filename = os.path.basename(localFilePath)
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            blob.download_to_filename(localFilePath)
        except Exception as e:
            return "ERROR: Error occurred in downloading file from cloud storage; error: {}".format(e)
        return True
    
    @staticmethod
    def deleteFile(filename):
        '''Returns True upon successful deletion.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            blob.delete()
        except Exception as e:
            return "ERROR: Error occurred in deleting file from cloud storage; error: {}".format(e)
        return True
    
    @staticmethod
    def deleteFiles(filenames: List[str]):
        '''Deletes multiple files from Firebase Storage. Returns True upon successful deletion.'''
        if not FireStorage.checkPermissions():
            return "ERROR: FireStorage service operation permission denied."
        try:
            bucket = storage.bucket()
            bucket.delete_blobs(filenames)
        except Exception as e:
            return "ERROR: Error occurred in deleting files from cloud storage; error: {}".format(e)
        return True