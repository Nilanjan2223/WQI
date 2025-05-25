import sys 
from logger import logging

def error_message_details(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script: [{0}] at line number: [{1}] with message: [{2}]".format(
        filename,exc_tb.tb_lineno,str(error))
    
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details=error_details)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Divide by zero error occurred")
        raise CustomException(e,sys) from e
    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
        print(ce)
        