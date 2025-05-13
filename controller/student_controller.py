from model.student import Student
from model.database import Database
from model.subject import Subject

class StudentController:
    def __init__(self, database: Database):
        self.database = database
        self.current_student = None
    
    def register(self, name: str, email: str, password: str) -> tuple[bool, str]:
        student = Student(name, email, password)
        
        if not student.validate_email():
            return False, "Invalid email format. Email must end with @university.com"
        
        if not student.validate_password():
            return False, "Invalid password format. Password must start with an uppercase letter, contain at least 5 letters, followed by at least 3 digits"
        
        if self.database.add_student(student):
            return True, "Registration successful"
        return False, "Email already registered"
    
    def login(self, email: str, password: str) -> tuple[bool, str]:
        student = self.database.get_student_by_email(email)
        if not student:
            return False, "Email not registered"
        
        if student.password != password:
            return False, "Pasword error"
        
        self.current_student = student
        return True, "Student login successfully"
    
    def change_password(self, new_password: str) -> tuple[bool, str]:
        if not self.current_student:
            return False, "Please login first"
        
        if not self.current_student.change_password(new_password):
            return False, "Invalid password format"
        
        if self.database.update_student(self.current_student):
            return True, "Password changed successfully"
        return False, "Password change failed" 
    
    def enroll_subject(self) -> tuple[bool, str]:
        if not self.current_student:
            return False, "Please login first"
        
        if len(self.current_student.subjects) >= 4:
            return False, "Maximum number of subjects reached"
        
        subject = Subject()
        if self.current_student.add_subject(subject):
            if self.database.update_student(self.current_student):
                return True, f"Course selection successful, Course ID: {subject.id}"
        return False, "Course selection failed"
    
    def remove_subject(self, subject_id: str) -> tuple[bool, str]:
        if not self.current_student:
            return False, "Please login first"
        
        if self.current_student.remove_subject(subject_id):
            if self.database.update_student(self.current_student):
                return True, "Course withdrawal successful"
        return False, "Course withdrawal failed, course not found"
    
    def show_enrollment(self) -> tuple[bool, str]:
        if not self.current_student:
            return False, "Please login first"
        
        if not self.current_student.subjects:
            return True, "No courses currently enrolled"
        
        result = "Current enrollment status:\n"
        for subject in self.current_student.subjects:
            result += f"Course ID: {subject.id}, Score: {subject.mark}, Grade: {subject.grade}\n"
        result += f"Average score: {self.current_student.get_average_mark():.2f}"
        return True, result 