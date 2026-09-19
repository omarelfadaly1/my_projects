# Libraries
import config 
from gui.app import FireAlertApp

def main():
    config.ensure_directories()
    app = FireAlertApp()
    app.mainloop()

if __name__ == "__main__":
    main()     