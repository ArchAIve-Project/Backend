import os
from importlib.metadata import distributions

class EnvVariable:
    def __init__(self, name: str, booleanValue: bool, required: bool=True, children: 'list[EnvVariable]'=None):
        self.name = name
        self.booleanValue = booleanValue
        self.required = required
        self.children = children if children is not None else []
    
    def check(self):
        if self.name not in os.environ and self.required:
            raise Exception("BOOTCHECK ERROR: Required environment variable '{}' was not found.".format(self.name))
        
        if self.booleanValue and os.environ.get(self.name, "False") == "True":
            if len(self.children) > 0:
                for child in self.children:
                    child.check()
        
        return self.name if (self.name not in os.environ and not self.required) else True
    
    @staticmethod
    def defaults():
        return [
            EnvVariable("DI_FAILOVER_STRATEGY", False, False),
            EnvVariable("DEVICE", False, False),
            EnvVariable("LOGGING_ENABLED", True, False),
            EnvVariable("DEBUG_MODE", True, False),
            EnvVariable("LLM_INFERENCE", True, True),
            EnvVariable("FM_DEBUG_MODE", True, False),
            EnvVariable("DECORATOR_DEBUG_MODE", True, False),
            EnvVariable("DB_DEBUG_MODE", True, False),
            EnvVariable("ARCHSMITH_ENABLED", True, True),
            EnvVariable("API_KEY", False, True),
            EnvVariable("SECRET_KEY", False, True),
            EnvVariable("FRONTEND_URL", False, True),
            EnvVariable("RAW_FILE_SERVICE", True, True),
            EnvVariable("MAX_FILE_UPLOADS", False, False),
            EnvVariable("MAX_ARTEFACT_SIZE", False, False),
            EnvVariable("MAX_PFP_SIZE", False, False),
            EnvVariable("MAX_CONTENT_SIZE", False, False),
            EnvVariable("EMAILING_ENABLED", True, True, [
                EnvVariable("SENDER_EMAIL", False, True),
                EnvVariable("SENDER_EMAIL_APP_PASSWORD", False, True)
            ]),
            EnvVariable("FireConnEnabled", True, True, [
                EnvVariable("FireRTDBEnabled", True, False, [
                    EnvVariable("RTDB_URL", False, True)
                ]),
                EnvVariable("FireStorageEnabled", True, False, [
                    EnvVariable("STORAGE_URL", False, True)
                ])
            ]),
            EnvVariable("LLMINTERFACE_ENABLED", True, True, [
                EnvVariable("QWEN_API_KEY", False, True),
                EnvVariable("OPENAI_API_KEY", False, True)
            ])
        ]

class BootCheck:
    dependencyMappings = {
        "python-dotenv": "python-dotenv",
        "requests": "requests",
        "torch": "torch",
        "torchvision": "torchvision",
        "flask": "Flask",
        "flask-limiter": "Flask-Limiter",
        "flask-cors": "flask-cors",
        "openai": "openai",
        "passlib": "passlib",
        "apscheduler": "APScheduler",
        "firebase-admin": "firebase-admin",
        "pillow": "pillow",
        "numpy": "numpy",
        "facenet-pytorch": "facenet-pytorch",
        "uuid": "uuid",
        "nltk": "nltk",
        "Pympler": "Pympler"
    }
    
    @staticmethod
    def getInstallations() -> list:
        pkgs = []
        for x in distributions():
            pkgs.append(x.name)
        return pkgs
    
    @staticmethod
    def checkDependencies():
        requiredDependencies = list(BootCheck.dependencyMappings.values())
        
        deps = BootCheck.getInstallations()
        for req in requiredDependencies:
            if req not in deps:
                raise Exception("BOOTCHECK ERROR: Required package '{}' not found.".format(req))
        
        return True
    
    @staticmethod
    def checkEnvVariables():
        from dotenv import load_dotenv
        load_dotenv()
        
        notFound = []
        for var in EnvVariable.defaults():
            res = var.check()
            if res != True:
                notFound.append(res)
        
        if len(notFound) > 0:
            print("BOOTCHECK WARNING: Optional environment variables {} not found.".format(', '.join(notFound)))
        
        return True
    
    @staticmethod
    def check():
        return BootCheck.checkDependencies() and BootCheck.checkEnvVariables()