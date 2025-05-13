import random

class Subject:
    def __init__(self):
        self.id = f"{random.randint(1, 999):03d}" # generate a random number between 1-999
        self.mark = random.randint(25, 100) # generate a random number between 25-100
        self.grade = self._calculate_grade()
    
    def _calculate_grade(self) -> str:
        if self.mark >= 85:
            return "HD"
        elif self.mark >= 75:
            return "D"
        elif self.mark >= 65:
            return "C"
        elif self.mark >= 50:
            return "P"
        else:
            return "F"
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'mark': self.mark,
            'grade': self.grade
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Subject':
        subject = cls()
        subject.id = data['id']
        subject.mark = data['mark']
        subject.grade = data['grade']
        return subject 