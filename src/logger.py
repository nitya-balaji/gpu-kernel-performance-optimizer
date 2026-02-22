#logging.py basically tells you everything that is happening right before the program breaks and also provides general info of what's happening at certain points in your code
import logging 
import os 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #creats a filename based on the current clock (to have unique files for each log)
logs_path = os.path.join(os.getcwd(), "logs") #gets the folder that we are currently in (project root) and builds a path for our logs folder using that
os.makedirs(logs_path,exist_ok=True) #"physically" creates the directory (folder) in the computer, exist_ok=True -> if the folder is already present, keep going

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) #combines the folder path and filename to create the exact "address" where the log text will be written

logging.basicConfig(
    filename=LOG_FILE_PATH, #tells us where to save the file
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", #tells us how the text should look (the time, line number where the log was triggered, whether its "info", "warning", or "error")
    level=logging.INFO, #tells Python to record everything that is "info" levl or higher, a brief message on what exactly happened)

)

if __name__ == "__main__":
    logging.info("Logging has started")
    print("Logger ran successfully!")