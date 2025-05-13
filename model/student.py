import random
import re
from typing import List
from .subject import Subject

class Student:
    def __init__(self, name: str, email: str, password: str):
        self.id = f"{random.randint(1, 999999):06d}"
        self.name = name
        self.email = email
        self.password = password
        self.subjects: List[Subject] = []
        
    def validate_email(self) -> bool:
        pattern = r'^[a-zA-Z0-9._]+@university\.com$'
        return bool(re.match(pattern, self.email))
    
    def validate_password(self) -> bool:
        pattern = r'^[A-Z][a-zA-Z]{4,}\d{3,}$'
        return bool(re.match(pattern, self.password))
    
    def add_subject(self, subject: Subject) -> bool:
        if len(self.subjects) >= 4:
            return False
        self.subjects.append(subject)
        return True
    
    def remove_subject(self, subject_id: str) -> bool:
        for i, subject in enumerate(self.subjects):
            if subject.id == subject_id:
                self.subjects.pop(i)
                return True
        return False
    
    def get_average_mark(self) -> float:
        if not self.subjects:
            return 0.0
        return sum(subject.mark for subject in self.subjects) / len(self.subjects)
    
    def has_passed(self) -> bool:
        return self.get_average_mark() >= 50
    
    def change_password(self, new_password: str) -> bool:
        if not re.match(r'^[A-Z][a-zA-Z]{4,}\d{3,}$', new_password):
            return False
        self.password = new_password
        return True
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'password': self.password,
            'subjects': [subject.to_dict() for subject in self.subjects]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Student':
        student = cls(data['name'], data['email'], data['password'])
        student.id = data['id']
        student.subjects = [Subject.from_dict(subject_data) for subject_data in data['subjects']]
        return student 