from typing import Tuple
from abc import ABC, abstractmethod

class EmailTemplate(ABC):
    @abstractmethod
    def generateDispatchParameters(self, *args, **kwargs) -> 'Tuple[str, str, str, str]':
        """Generate the parameters needed to dispatch the email.

        Returns: `Tuple[str, str, str, str]`: The destination email, subject, text content, and HTML content.
        """
        pass