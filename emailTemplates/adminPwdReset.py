from services import Universal
from schemas import User
from .baseTemplate import EmailTemplate
from .jinjaEnv import env as TemplatesEnv

class AdminPasswordResetAlert(EmailTemplate):
    text = """Dear {},
A system admin has reset your password.
Please login with your new password: {}

System admins typically carry out this action in the event you've forgotten your password, or due to some other security issue.
Should you require more clarification or assistance, please contact the system administrator.
Thank you for being a valued user of ArchAIve.

This is a system generated email delivered by ArchAIve.
{}"""
    
    def __init__(self, user: User, newPwd: str):
        self.destEmail = user.email
        self.subject = "Admin Password Reset | ArchAIve"
        self.text = AdminPasswordResetAlert.generateText(user.fname, newPwd)
        self.html = TemplatesEnv.get_template('emails/adminPwdReset.html').render(fname=user.fname, newPwd=newPwd, copyright=Universal.copyright)
    
    def generateDispatchParameters(self):
        return self.destEmail, self.subject, self.text, self.html
    
    @staticmethod
    def generateText(fname: str, newPwd: str) -> str:
        return AdminPasswordResetAlert.text.format(fname, newPwd, Universal.copyright)