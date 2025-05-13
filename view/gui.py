import tkinter as tk
from tkinter import ttk, messagebox
from model.database import Database
from controller.student_controller import StudentController

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("University System")
        self.root.geometry("400x300")
        
        self.database = Database()
        self.student_controller = StudentController(self.database)
        
        self.create_login_window()
    
    def create_login_window(self):
        self.clear_window()
        
        ttk.Label(self.root, text="Student Login", font=("Arial", 16)).pack(pady=20)
        
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        
        ttk.Label(frame, text="Email:").grid(row=0, column=0, padx=5, pady=5)
        self.email_entry = ttk.Entry(frame)
        self.email_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Password:").grid(row=1, column=0, padx=5, pady=5)
        self.password_entry = ttk.Entry(frame, show="*")
        self.password_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(self.root, text="Login", command=self.login).pack(pady=10)
    
    def create_enrollment_window(self):
        self.clear_window()
        
        ttk.Label(self.root, text="Course Enrollment System", font=("Arial", 16)).pack(pady=20)
        
        # Display current enrollments
        self.enrollment_text = tk.Text(self.root, height=10, width=40)
        self.enrollment_text.pack(pady=10)
        self.update_enrollment_display()
        
        # Enroll button
        ttk.Button(self.root, text="Enroll", command=self.enroll_subject).pack(pady=5)
        
        # Logout button
        ttk.Button(self.root, text="Logout", command=self.logout).pack(pady=5)
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        
        if not email or not password:
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        success, message = self.student_controller.login(email, password)
        if success:
            self.create_enrollment_window()
        else:
            messagebox.showerror("Error", message)
    
    def enroll_subject(self):
        if len(self.student_controller.current_student.subjects) >= 4:
            messagebox.showerror("Error", "Maximum number of enrolled subjects reached (4)")
            return
        
        success, message = self.student_controller.enroll_subject()
        if success:
            messagebox.showinfo("Success", message)
            self.update_enrollment_display()
        else:
            messagebox.showerror("Error", message)
    
    def update_enrollment_display(self):
        self.enrollment_text.delete(1.0, tk.END)
        success, message = self.student_controller.show_enrollment()
        self.enrollment_text.insert(tk.END, message)
    
    def logout(self):
        self.student_controller.current_student = None
        self.create_login_window()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = GUI()
    gui.run()
