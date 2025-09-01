import sys
import logging

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message with filename, line number, and error description.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in file: {file_name} at line: {line_number} - {str(error)}"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logging.error(self.error_message) 

    def __str__(self):
        return self.error_message


