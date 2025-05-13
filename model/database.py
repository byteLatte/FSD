import json
import os
from typing import List
from model.student import Student


class Database:
    def __init__(self, filename: str = "students.data"):
        self.filename = filename
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                json.dump([], f)
    
    def read_students(self) -> List[Student]:
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return [Student.from_dict(student_data) for student_data in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def write_students(self, students: List[Student]):
        with open(self.filename, 'w') as f:
            json.dump([student.to_dict() for student in students], f, indent=4)
    
    def clear_database(self):
        with open(self.filename, 'w') as f:
            json.dump([], f)
    
    def add_student(self, student: Student) -> bool:
        students = self.read_students()
        if any(s.email == student.email for s in students):
            return False
        students.append(student)
        self.write_students(students)
        return True
    
    def remove_student(self, student_id: str) -> bool:
        students = self.read_students()
        for i, student in enumerate(students):
            if student.id == student_id:
                students.pop(i)
                self.write_students(students)
                return True
        return False
    
    def get_student_by_email(self, email: str) -> Student:
        students = self.read_students()
        for student in students:
            if student.email == email:
                return student
        return None
    
    def update_student(self, updated_student: Student) -> bool:
        students = self.read_students()
        for i, student in enumerate(students):
            if student.id == updated_student.id:
                students[i] = updated_student
                self.write_students(students)
                return True
        return False 