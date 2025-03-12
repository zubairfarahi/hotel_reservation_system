import datetime
import logging

class CustomFormatter(logging.Formatter):
    def __init__(self):
        # Example of how the fmt prints:
        # INFO       2022/01/01 12:34:56.789000 called from home/zlogger/custom_formatter.py, line 10, function __init__, module_name custom_formatter, requestID: 12345; Some additional data This is a log message
        super().__init__(
            fmt='%(levelname)-10s %(asctime1)s called from %(file_path)s, line %(line_no)s, function %(function_name)s, module_name %(module_name)s, requestID: %(request_id)s; %(data)s %(message)s',
        )

    @staticmethod
    def formatTime(self, datefmt=None):
        # Example: format timestamp as "2022/01/01 12:34:56.789000"
        timestamp = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp())
        microseconds = timestamp.microsecond // 1000
        formatted_timestamp = f"{timestamp:%Y/%m/%d %H:%M:%S}.{microseconds:06d}"
        return formatted_timestamp