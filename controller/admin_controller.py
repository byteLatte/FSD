from typing import List, Dict
from model.student import Student
from model.database import Database

class AdminController:
    def __init__(self, database: Database):
        self.database = database
    
    def clear_database(self) -> tuple[bool, str]:
        choice = input("\t\tAre you sure you want to clear the database (Y)ES/(N)o:").upper()
        if choice != "Y":
            return False, "\t\tdatabase clear operation cancelled"
        else:
            self.database.clear_database()
            print("\t\tDatabase cleared")
        return True, "database cleared" 
    
    def group_students_by_grade(self) -> tuple[bool, str]:
        students = self.database.read_students()
        if not students:
            return True, "No students data"
        
        grade_groups: Dict[str, List[Student]] = {
            "HD": [],
            "D": [],
            "C": [],
            "P": [],
            "F": []
        }
        
        for student in students:
            if not student.subjects:
                continue
            avg_mark = student.get_average_mark()
            if avg_mark >= 85:
                grade_groups["HD"].append(student)
            elif avg_mark >= 75:
                grade_groups["D"].append(student)
            elif avg_mark >= 65:
                grade_groups["C"].append(student)
            elif avg_mark >= 50:
                grade_groups["P"].append(student)
            else:
                grade_groups["F"].append(student)
        
        result = "\tGrade groups:\n"

        for grade, group in grade_groups.items():
            result += f"\t{grade} ({len(group)} students):\n"
            for student in group:
                result += f"\t\t  ID: {student.id}, Name: {student.name}, Average: {student.get_average_mark():.2f}\n"
        return True, result
    
    def partition_students(self) -> tuple[bool, str]:
        students = self.database.read_students()
        if not students:
            return True, "No students data"
        
        pass_students = []
        fail_students = []
        
        for student in students:
            if student.has_passed():
                pass_students.append(student)
            else:
                fail_students.append(student)
        
        result = "\tPASS/FAIL partition:\n"
        result += f"\tPASS ({len(pass_students)} students):\n"
        for student in pass_students:
            result += f"\t\t  ID: {student.id}, Name: {student.name}, Average: {student.get_average_mark():.2f}\n"
        
        result += f"\tFAIL ({len(fail_students)} students):\n"
        for student in fail_students:
            result += f"\t\t  ID: {student.id}, Name: {student.name}, Average: {student.get_average_mark():.2f}\n"
        
        return True, result
    
    def remove_student(self, student_id: str) -> tuple[bool, str]:
        if self.database.remove_student(student_id):
            return True, "\t\tStudent removed"
        return False, "\t\tStudent not found"
    
    def show_all_students(self) -> tuple[bool, str]:
        students = self.database.read_students()
        if not students:
            return True, "\t\tNo students data"
        
        result = "\tAll students:\n"
        for student in students:
            result += f"\tID: {student.id}, Name: {student.name}, Email: {student.email}\n"
            result += "\tSubjects:\n"
            for subject in student.subjects:
                result += f"\t\tSubject ID: {subject.id}, Mark: {subject.mark}, Grade: {subject.grade}\n"
            result += f"\tAverage: {student.get_average_mark():.2f}\n"
            result+="=="*20+"\n"
        return True, result 