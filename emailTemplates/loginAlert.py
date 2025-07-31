from models import User
from .baseTemplate import EmailTemplate
from .jinjaEnv import env as TemplatesEnv

class LoginAlert(EmailTemplate):
    text = """Dear {},
We noticed a login to your account on {}. If this was you, no further action is needed.
If you did not authorize this login, please secure your account immediately by changing your password and reviewing your account activity.
For any assistance, feel free to contact our support team.
Thank you for using ArchAIve!
Best regards,
The ArchAIve Team

This is an automated message. Please do not reply directly to this email."""
    
    def __init__(self, user: User):
        self.destEmail = user.email
        self.subject = "New Login | ArchAIve"
        self.text = LoginAlert.generateText(user.username, user.lastLogin)
        self.html = TemplatesEnv.get_template('emails/loginAlert.html').render(username=user.username, lastLogin=user.lastLogin)
    
    def generateDispatchParameters(self):
        return self.destEmail, self.subject, self.text, self.html
    
    @staticmethod
    def generateText(username: str, lastLogin: str) -> str:
        return LoginAlert.text.format(username, lastLogin)