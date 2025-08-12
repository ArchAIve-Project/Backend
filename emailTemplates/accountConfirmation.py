import os
from services import Universal
from schemas import User
from .baseTemplate import EmailTemplate
from .jinjaEnv import env as TemplatesEnv

class AccountConfirmationAlert(EmailTemplate):
    text = """Dear {},
歡迎來到 ArchAIve!
ArchAIve is an AI-powered artefact digitisation platform employed by your organisation.
The system administrator from your organisation has created an account for you to access the system's services!

To access your account, please login into the system at {} with the following credentials:
Username: {}
Password: {}

Should you have any questions/queries, please contact the system administrator.
A very warm welcome to ArchAIve; we hope you have a pleasant experience interacting with the system's innovative features.

謝謝，祝您有個愉快的一天

This is a system generated email delivered by ArchAIve.
{}"""
    
    def __init__(self, user: User, userPwd: str):
        self.destEmail = user.email
        self.subject = "Your New Account | ArchAIve"
        
        loginURL = os.environ.get("FRONTEND_URL", "https://archaive.com") + "/auth/login"
        
        self.text = AccountConfirmationAlert.generateText(
            fname=user.fname or user.username,
            loginURL=loginURL,
            username=user.username,
            password=userPwd
        )
        
        self.html = TemplatesEnv.get_template('emails/accountConfirmation.html').render(
            fname=user.fname or user.username,
            loginURL=loginURL,
            username=user.username,
            password=userPwd,
            copyright=Universal.copyright
        )
    
    def generateDispatchParameters(self):
        return self.destEmail, self.subject, self.text, self.html
    
    @staticmethod
    def generateText(fname: str, loginURL: str, username: str, password: str) -> str:
        return AccountConfirmationAlert.text.format(fname, loginURL, username, password, Universal.copyright)