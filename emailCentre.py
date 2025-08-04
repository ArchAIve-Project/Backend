from emailer import Emailer
from emailTemplates import *

class EmailCentre:
    @staticmethod
    def setup():
        Emailer.checkContext()
    
    @staticmethod
    def dispatch(template: EmailTemplate):
        if Emailer.servicesEnabled:
            return Emailer.sendEmail(*template.generateDispatchParameters())
        return False