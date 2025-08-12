from services import Logger
from emailer import Emailer
from emailTemplates import *

class EmailCentre:
    @staticmethod
    def setup():
        Emailer.checkContext()
    
    @staticmethod
    def dispatch(template: EmailTemplate):
        output = False
        if Emailer.servicesEnabled:
            output = Emailer.sendEmail(*template.generateDispatchParameters())
        
        Logger.log("EMAILCENTRE: Dispatch finished with result: {}".format(output))
        return output