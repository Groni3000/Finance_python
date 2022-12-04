from abc import ABC, abstractmethod


class Indicator(ABC):
    name:str = 'Unknown indicator'
    description:str = 'Unknown'
    
    @abstractmethod
    def run(self):
        ...
    