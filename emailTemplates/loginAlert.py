import datetime
from services import Universal
from schemas import User
from .baseTemplate import EmailTemplate
from .jinjaEnv import env as TemplatesEnv

class LoginAlert(EmailTemplate):
    text = """Dear {},
We noticed a login to your account on {}. If this was you, no further action is needed.
If you did not authorize this login, please secure your account immediately by changing your password and reviewing your account activity.

Thank you for being a valued user of ArchAIve.

This is a system generated email delivered by ArchAIve.
{}"""
    
    def __init__(self, user: User):
        self.destEmail = user.email
        self.subject = "New Login | ArchAIve"
        lastLogin = Universal.fromUTC(user.lastLogin, localisedTo=Universal.localisationOffset).strftime("%d %B, %A, %Y %H:%M:%S%p")
        self.text = LoginAlert.generateText(user.fname, lastLogin)
        self.html = TemplatesEnv.get_template('emails/loginAlert.html').render(fname=user.fname, lastLogin=lastLogin, copyright=Universal.copyright)
    
    def generateDispatchParameters(self):
        return self.destEmail, self.subject, self.text, self.html
    
    @staticmethod
    def generateText(fname: str, lastLogin: str) -> str:
        return LoginAlert.text.format(fname, lastLogin, Universal.copyright)