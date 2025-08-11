from services import Universal
from schemas import User
from .baseTemplate import EmailTemplate
from .jinjaEnv import env as TemplatesEnv

class PasswordChangedAlert(EmailTemplate):
    text = """Dear {},
This is a notification to alert you that, upon your request, your account password has been changed successfully.
If this was not you, please contact the system administrator immediately to reset your password to maintain account security.
A gentle reminder to keep passwords in a safe place, and to not share with anyone you may not trust.

Thank you for being a valued user of ArchAIve.

This is a system generated email delivered by ArchAIve.
{}"""
    
    def __init__(self, user: User):
        self.destEmail = user.email
        self.subject = "Password Changed | ArchAIve"
        self.text = PasswordChangedAlert.generateText(user.fname)
        self.html = TemplatesEnv.get_template('emails/changePwdAlert.html').render(fname=user.fname or user.username, copyright=Universal.copyright)
    
    def generateDispatchParameters(self):
        return self.destEmail, self.subject, self.text, self.html
    
    @staticmethod
    def generateText(fname: str) -> str:
        return PasswordChangedAlert.text.format(fname, Universal.copyright)