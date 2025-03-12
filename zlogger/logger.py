
import logging
import os
import inspect
from functools import wraps
import sys
from custom_formatter import CustomFormatter
from custom_file_rotater import CustomFileRotator
import time
from constants import *
import configparser


# Add custom log levels to the logging module
logging.addLevelName(CustomLogLevel.SUCCESS_LEVEL.value, SUCCESS)
logging.addLevelName(CustomLogLevel.REJECT_LEVEL.value, REJECT)
logging.addLevelName(CustomLogLevel.FATAL_LEVEL.value, FATAL)

class ZLogger(logging.Logger):
    def __init__(self, name, config, level=logging.INFO):
        super().__init__(name, level)
        self.configure_logger(config)
        self.extra_context = {}

    def validate_config(self, config):
        """
        Validate the logging configuration parameters.
        
        Parameters:
        config (ConfigParser): Configuration object containing logging settings.
        
        Returns:
        Validated logging settings or None if validation fails.
        """
        
        # Validate log level
        try:
            log_level = config.get(LogConfig.LOG.value, LogConfig.LEVEL.value).upper()

            if log_level not in LogLevel.list():
                logging.error(f"Invalid log level: {log_level}. Valid options are: {', '.join(LogLevel.list())}")

        except configparser.NoOptionError as e:
            logging.exception(e)
            

        # Validate log_stdout and log_stderr
        try:
            log_stdout = config.getboolean(LogConfig.LOG.value, LogConfig.LOG_STDOUT.value)
            log_stderr = config.getboolean(LogConfig.LOG.value, LogConfig.LOG_STDERR.value)
        except configparser.NoOptionError as e:
            logging.exception(e)
            
        except ValueError:
            logging.error(ERROR_DESC['312'])
            

        
        # Validate file logging configuration if enabled
        log_file_config = {}
        if config.getboolean(LogConfig.LOG_FILE.value, LogConfig.ENABLED.value):
            log_file_config = self._validate_log_file_config(config)
            if not log_file_config:
                return None

        return log_level, log_stdout, log_stderr, log_file_config

    def _validate_log_file_config(self, config):
        """
        Validate the file logging configuration parameters.
        
        Parameters:
        config (ConfigParser): Configuration object containing logging settings.
        
        Returns:
        dict: Validated file logging settings or None if validation fails.
        """
        
        log_file_config = {}

        # Validate log filename
        log_filename = config.get(LogConfig.LOG_FILE.value, LogConfig.FILE_NAME.value)
        if not log_filename:
            logging.error(ERROR_DESC['303'])

        log_file_config[LogConfig.FILE_NAME.value] = log_filename

        # Validate log file extension
        log_file_extension = config.get(LogConfig.LOG_FILE.value, LogConfig.FILE_EXTENSION.value)
        if not log_file_extension:
            logging.error(ERROR_DESC['305'])

        log_file_config[LogConfig.FILE_EXTENSION.value] = log_file_extension

        # Validate log file path
        log_file_path = config.get(LogConfig.LOG_FILE.value, LogConfig.LOG_PATH.value)
        if not log_file_path:
            logging.error(ERROR_DESC['310'])
            
        log_file_config[LogConfig.LOG_PATH.value] = log_file_path

        # Validate log file size, age, and storage settings
        log_max_file_size = config.getint(LogConfig.LOG_FILE.value, LogConfig.MAX_FILE_SIZE.value)
        if not config.get(LogConfig.LOG_FILE.value, LogConfig.MAX_FILE_SIZE.value):
            logging.error(ERROR_DESC['311'])

        log_file_config[LogConfig.MAX_FILE_SIZE.value] = log_max_file_size

        log_max_age_days = config.getint(LogConfig.LOG_FILE.value, LogConfig.MAX_AGE_DAYS.value)
        if not config.get(LogConfig.LOG_FILE.value, LogConfig.MAX_AGE_DAYS.value):
            logging.error(ERROR_DESC['313'])

        log_file_config[LogConfig.MAX_AGE_DAYS.value] = log_max_age_days

        log_max_storage_size = config.getint(LogConfig.LOG_FILE.value, LogConfig.MAX_STORAGE_SIZE.value)
        if not config.get(LogConfig.LOG_FILE.value, LogConfig.MAX_STORAGE_SIZE.value):
            logging.error(ERROR_DESC['314'])

        log_file_config[LogConfig.MAX_STORAGE_SIZE.value] = log_max_storage_size

        log_archive_path = config.get(LogConfig.LOG_FILE.value, LogConfig.ARCHIVE_PATH.value)
        if not config.get(LogConfig.LOG_FILE.value, LogConfig.ARCHIVE_PATH.value):
            logging.error(ERROR_DESC['315'])

        log_file_config[LogConfig.ARCHIVE_PATH.value] = log_archive_path

        # Create the log file path if it doesn't exist
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
            
        # Create the archive log file path if it doesn't exist
        if not os.path.exists(log_archive_path):
            os.makedirs(log_archive_path)

        # Append timestamp to the log filename
        complete_file_name = log_filename + log_file_extension + time.strftime('-%Y-%m-%d-%H%M%S')
        log_file_path = os.path.join(log_file_path, complete_file_name)
        log_file_config[LogConfig.LOG_FILE_PATH.value] = log_file_path

        return log_file_config

    def configure_logger(self, config):
        """
        Configure the logger based on the provided configuration.
        
        Parameters:
        config (ConfigParser): Configuration object containing logging settings.
        """
        
        log_level, log_stdout, log_stderr, log_file_config = self.validate_config(config)

        formatter = CustomFormatter()
        handlers = self._create_handlers(log_level, log_stdout, log_stderr, log_file_config, formatter)
        self._configure_loggers(log_level, handlers)

    def _create_handlers(self, log_level, log_stdout, log_stderr, log_file_config, formatter):
        """
        Create logging handlers based on configuration settings.
        
        Parameters:
        log_level (str): The log level for the handlers.
        log_stdout (bool): Whether to log to stdout.
        log_stderr (bool): Whether to log to stderr.
        log_file_config (dict): File logging configuration.
        formatter (logging.Formatter): The formatter for the handlers.
        
        Returns:
        list: A list of logging handlers.
        """
        
        handlers = []
        if log_stdout:
            handlers.append(self._create_console_handler(log_level, formatter, sys.stdout))

        if log_stderr:
            handlers.append(self._create_console_handler(logging.ERROR, formatter, sys.stderr))

        if log_file_config:
            handlers.append(self._create_file_handler(log_level, formatter, log_file_config))

        return handlers

    def _create_console_handler(self, log_level, formatter, stream):
        """
        Create a console logging handler.
        
        Parameters:
        log_level (str): The log level for the handler.
        formatter (logging.Formatter): The formatter for the handler.
        stream (io.TextIOWrapper): The stream (stdout or stderr) for the handler.
        
        Returns:
        logging.StreamHandler: The console logging handler.
        """
        
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(logging.getLevelName(log_level))
        console_handler.setFormatter(formatter)
        return console_handler

    def _create_file_handler(self, log_level, formatter, log_file_config):
        """
        Create a file logging handler with rotation capabilities.
        
        Parameters:
        log_level (str): The log level for the handler.
        formatter (logging.Formatter): The formatter for the handler.
        log_file_config (dict): File logging configuration.
        
        Returns:
        CustomFileRotator: The file logging handler.
        """
        
        custom_file_handler = CustomFileRotator(
            log_file_config[LogConfig.FILE_NAME.value],
            log_file_config[LogConfig.FILE_EXTENSION.value],
            log_path=log_file_config[LogConfig.LOG_PATH.value],
            filename=log_file_config[LogConfig.LOG_FILE_PATH.value],
            max_file_size=log_file_config[LogConfig.MAX_FILE_SIZE.value],
            max_age_days=log_file_config[LogConfig.MAX_AGE_DAYS.value],
            max_storage_size=log_file_config[LogConfig.MAX_STORAGE_SIZE.value],
            archive_path=log_file_config[LogConfig.ARCHIVE_PATH.value]
        )
        custom_file_handler.setLevel(logging.getLevelName(log_level))
        custom_file_handler.setFormatter(formatter)
        return custom_file_handler

    def _configure_loggers(self, log_level, handlers):
        """
        Configure the logger with the specified handlers and log level.
        
        Parameters:
        log_level (str): The log level for the logger.
        handlers (list): A list of logging handlers.
        """
        
        self.setLevel(log_level)
        for handler in handlers:
            self.addHandler(handler)
        
    def with_request_id(self, request_id):
        """
        Add request ID to the logger's extra context.
        
        Parameters:
        request_id (str): The request ID to be added.
        
        Returns:
        ZLogger: The logger instance with updated context.
        """
        
        self.extra_context[LogConfig.REQUEST_ID.value] = request_id
        return self
    
    def with_module_name(self, module_name):
        """
        Add module name to the logger's extra context.
        
        Parameters:
        moduld_name (str): The module name to be added.
        
        Returns:
        ZLogger: The logger instance with updated context.
        """
        
        self.extra_context[LogConfig.MODULE_NAME.value] = module_name
        return self
    
    def with_additional_data(self, additional_data):
        """
        Add additional data to the logger's extra context.
        
        Parameters:
        additional_data (dict): A dictionary of additional data to be added.
        
        Returns:
        ZLogger: The logger instance with updated context.
        """
        
        if not self.extra_context:
            self.extra_context = additional_data.copy()
        else:
            self.extra_context.update(additional_data)
        return self

    def log_decorator(func):
        """
        A decorator for logging methods to add extra context to log records.
        
        Parameters:
        func (function): The logging method to be decorated.
        
        Returns:
        function: The wrapped logging method with extra context.
        """
        
        @wraps(func)
        def wrapper(self, message, *args, **kwargs):
            caller_frame = inspect.currentframe().f_back
            caller_func_name = caller_frame.f_code.co_name
            caller_file_name = caller_frame.f_code.co_filename
            line_no = caller_frame.f_lineno

            if caller_func_name == "<module>":
                caller_func_name = os.path.splitext(os.path.basename(caller_file_name))[0]

            data = {k: v for k, v in self.extra_context.items() if k not in [LogConfig.REQUEST_ID.value, LogConfig.MODULE_NAME.value]}

            extra_context = {
                LogConfig.REQUEST_ID.value: self.extra_context.get(LogConfig.REQUEST_ID.value),
                LogConfig.FUNCTION_NAME.value: caller_func_name,
                LogConfig.FILE_PATH.value: caller_file_name,
                LogConfig.LINE_NO.value: '#' + str(line_no),
                LogConfig.ASCTIME1.value: CustomFormatter.formatTime(self),
                LogConfig.DATA.value: ' '.join(f"{k}: {v}," for k, v in data.items()) if data else '',
                LogConfig.MODULE_NAME.value: self.extra_context.get(LogConfig.MODULE_NAME.value)
            }

            self.extra_context = {}
            
            return func(self, message, *args, extra=extra_context)
        return wrapper

    @log_decorator
    def fatal(self, message, *args, **kwargs):
        """Log a message with FATAL level."""
        self.log(CustomLogLevel.FATAL_LEVEL.value, message, *args, **kwargs)

    @log_decorator
    def reject(self, message, *args, **kwargs):
        """Log a message with REJECT level."""
        self.log(CustomLogLevel.REJECT_LEVEL.value, message, *args, **kwargs)

    @log_decorator
    def success(self, message, *args, **kwargs):
        """Log a message with SUCCESS level."""
        self.log(CustomLogLevel.SUCCESS_LEVEL.value, message, *args, **kwargs)

    @log_decorator
    def debug(self, message, *args, **kwargs):
        """Log a message with DEBUG level."""
        self.log(logging.DEBUG, message, *args, **kwargs)

    @log_decorator
    def info(self, message, *args, **kwargs):
        """Log a message with INFO level."""
        self.log(logging.INFO, message, *args, **kwargs)

    @log_decorator
    def warning(self, message, *args, **kwargs):
        """Log a message with WARNING level."""
        self.log(logging.WARNING, message, *args, **kwargs)

    @log_decorator
    def error(self, message, *args, **kwargs):
        """Log a message with ERROR level."""
        self.log(logging.ERROR, message, *args, **kwargs)