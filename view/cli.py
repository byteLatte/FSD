from model.database import Database
from controller.student_controller import StudentController
from controller.admin_controller import AdminController

# Command line interface for the university system
class CLI:
    def __init__(self):
        self.database = Database()
        self.student_controller = StudentController(self.database)
        self.admin_controller = AdminController(self.database)
    
    def run(self):
        while True:
            choice = input("University System: (A)dmin, (S)tudent, or X:").upper()
            
            if choice == "A":
                self.admin_menu()
            elif choice == "S":
                self.student_menu()
            elif choice == "X":
                print("Thank you")
                break
            else:
                print("Invalid choice")
    
    def admin_menu(self):
        while True:
            choice = input("\tAdmin System: (c)lear, (g)roup, (p)artition, (r)emove, (s)how, or X:").lower()
            
            if choice == "c":
                success, message = self.admin_controller.clear_database()
                print(message)
            elif choice == "g":
                success, message = self.admin_controller.group_students_by_grade()
                print(message)
            elif choice == "p":
                success, message = self.admin_controller.partition_students()
                print(message)
            elif choice == "r":
                student_id = input("Remove student ID:")
                success, message = self.admin_controller.remove_student(student_id)
                print(message)
            elif choice == "s":
                success, message = self.admin_controller.show_all_students()
                print(message)
            elif choice == "x":
                break
            else:
                print("Invalid choice")

    def student_menu(self):
        while True:
            choice = input("Student System (l)ogin/(r)egister/(x)exit: ").lower()
            print("\tStudent Sign IN")
            if choice == "l":
                email = input("\tEmail: ")
                password = input("\tPassword: ")
                success, message = self.student_controller.login(email, password)
                print(message)
                if success:
                    self.enrollment_menu()
            elif choice == "r":
                name = input("Name: ")
                email = input("Email: ")
                password = input("Password: ")
                success, message = self.student_controller.register(name, email, password)
                print(message)
            elif choice == "x":
                break
            else:
                print("Invalid choice")

    def enrollment_menu(self):
        while True:
            print("\nCourse Enrollment System")
            print("(c) Change Password")
            print("(e) Enroll in Course")
            print("(r) Remove Course")
            print("(s) Show Enrolled Courses")
            print("(x) Exit")
            
            choice = input("Please choose: ").lower()
            
            if choice == "c":
                new_password = input("New Password: ")
                success, message = self.student_controller.change_password(new_password)
                print(message)
            elif choice == "e":
                success, message = self.student_controller.enroll_subject()
                print(message)
            elif choice == "r":
                subject_id = input("Course ID: ")
                success, message = self.student_controller.remove_subject(subject_id)
                print(message)
            elif choice == "s":
                success, message = self.student_controller.show_enrollment()
                print(message)
            elif choice == "x":
                self.student_controller.current_student = None
                break
            else:
                print("Invalid choice")

if __name__ == "__main__":
    cli = CLI()
    cli.run() 

















'''

CLI类是一个命令行界面类，用于实现大学管理系统的用户交互界面。该类集成了学生管理和管理员管理功能。
属性
database: Database实例，用于数据存储
student_controller: StudentController实例，处理学生相关操作
admin_controller: AdminController实例，处理管理员相关操作

方法
__init__()
功能：初始化CLI类
参数：无
说明：创建数据库连接并初始化学生和管理员控制器
run()
功能：运行主程序循环
参数：无
说明：
显示主菜单选项：管理员(A)、学生(S)、退出(X)
根据用户输入调用相应的功能模块
循环执行直到用户选择退出
admin_menu()
功能：管理员功能菜单
参数：无
说明：提供以下功能选项：
(c) 清空数据库
(g) 按成绩分组
(p) 学生分类
(r) 删除学生
(s) 显示所有学生
(x) 退出
student_menu()
功能：学生功能菜单
参数：无
说明：提供以下功能选项：
(l) 登录
(r) 注册
(x) 退出
enrollment_menu()
功能：选课系统菜单
参数：无
说明：提供以下功能选项：
(c) 修改密码
(e) 选课
(r) 退课
(s) 显示选课
(x) 退出

'''