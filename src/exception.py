import sys
import types


def error_message_detail(error: Exception, error_detail: types.ModuleType) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
    line_number = exc_tb.tb_lineno if exc_tb else "<unknown>"
    return (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] error message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: types.ModuleType):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
