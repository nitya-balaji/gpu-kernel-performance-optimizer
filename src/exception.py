import sys  
from src.logger import logging #this will create a log file (based on steps outlined in logger.py script) for an raised errors (if not it will just raise the custom exception in the terminal)

#function that formats the error message 
def error_message_detail(error,error_detail:sys): #error = actual error, error_detail = details about the error which can be found bc we imported the sys module (it will grab the details of the error that is stored in Python's memory)
    _,_,exc_tb=error_detail.exc_info() #exc_info() returns three things: type, value, and traceback -> we only care about traceback (so we use _ to ignore the first two) - exec_tb is the object that contains the info about where exactly the code failed
    file_name=exc_tb.tb_frame.f_code.co_filename #finding the file name of where the error came from by looking into the exc_tb object
    error_message="Error occurred in Python script name [{0}], line number [{1}], with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message

#the above function only creates the error message, but we need a class to actually "raise" the error
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message) #this tells Python that we will be adding our own custom message (in addition to the normal error features)
        self.error_message = error_message_detail(error_message, error_detail=error_detail) #initialize the error message that we want based on the function we created above
    
    def __str__(self): #this will show our well-formatted error message (when the error is raised our custom error message will be printed with this method)
        return self.error_message 