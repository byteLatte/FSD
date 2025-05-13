import sys
from view.cli import CLI
from view.gui import GUI

def main():
    print("Please choose the running mode:")
    print("1. Command Line Interface (CLI)")
    print("2. Graphical User Interface (GUI)")
    
    choice = input("Please enter the option (1/2): ")
    
    if choice == "1":
        cli = CLI()
        cli.run()
    elif choice == "2":
        gui = GUI()
        gui.run()
    else:
        print("Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main() 