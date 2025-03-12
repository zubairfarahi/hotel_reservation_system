import os
import time
import shutil
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta

class CustomFileRotator(TimedRotatingFileHandler):
    def __init__(self,name, file_extension,  filename, log_path, max_file_size, max_age_days, max_storage_size, archive_path=None, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False, atTime=None):
        """
        Initialize the CustomFileRotator handler.

        Parameters:
        name (str): The base name for the log files.
        file_extension (str): The extension for the log files.
        filename (str): The initial log file name.
        log_path (str): The directory path where logs are stored.
        max_file_size (int): The maximum size (in bytes) before a log file is rotated.
        max_age_days (int): The maximum age (in days) before a log file is archived.
        max_storage_size (int): The maximum storage size (in bytes) for all log files before archiving.
        archive_path (str): The path where archived log files are stored.
        when (str): The type of interval (e.g., 'h' for hours).
        interval (int): The interval at which to rotate the log file.
        backupCount (int): The number of backup files to keep.
        encoding (str): The encoding to use for the log files.
        delay (bool): Whether to delay the creation of the log file.
        utc (bool): Whether to use UTC for time calculations.
        atTime (datetime.time): The time at which to perform the log rotation.
        """
        
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)
        self.name = name
        self.file_extension = file_extension
        self.log_path = log_path
        self.max_file_size = max_file_size
        self.max_age_days = max_age_days
        self.max_storage_size = max_storage_size
        self.archive_path = archive_path
        self.rolloverAt = self.computeRollover(int(time.time()))
        
        # If an archive path is specified, archive all log files in the current directory except the latest one.
        if self.archive_path:
            current_directory = os.path.dirname(self.baseFilename)
            files = sorted(os.listdir(current_directory))
            for file in files[:-1]:
                file_path = os.path.join(current_directory, file)
                if os.path.isfile(file_path):
                    shutil.copy2(file_path, os.path.join(self.archive_path, file))
    
    
    def get_size(self, path):
        """
        Calculate the total size of files in a directory.

        Parameters:
        path (str): The directory path.

        Returns:
        int: The total size of files in bytes.
        """
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
        
    def computeRollover(self, currentTime):
        """
        Compute the time at which the next rollover should occur.

        Parameters:
        currentTime (int): The current time in seconds since epoch.

        Returns:
        int: The timestamp for the next rollover.
        """
        
        if self.atTime is None:
            # If atTime is not provided, use the current time for midnight rotation
            return (currentTime // 86400 + 1) * 86400
        else:
            return self.atTime.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def shouldRollover(self, record):
        """
        Determine if the log file should be rolled over.

        Parameters:
        record (LogRecord): The log record that is being processed.

        Returns:
        bool: True if rollover should occur, False otherwise.
        """
        
        if self.stream is None:
            self.stream = self._open()
        if self.max_file_size > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self.max_file_size:
                return True
      
        # Check if the log path has reached the maximum storage size.
        # If it has, remove the oldest log file one at a time until the size is below the limit.
        if self.get_size(self.log_path) >= self.max_storage_size:
            curent_directory = os.path.dirname(self.baseFilename)
            files = sorted(os.listdir(self.log_path))
            
            file_archive_path = os.path.join(self.archive_path, files[1])            
            file_path = os.path.join(curent_directory, files[1])
            if os.path.isfile(file_archive_path):
                os.remove(file_path)
            else:
                shutil.copy2(file_path, self.archive_path)
                os.remove(file_path)
                
        return False
    
    def doRollover(self):
        """
        Perform the rollover of the log file.
        """
        
        if self.stream:
            self.stream.close()
            self.stream = None
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        
        # At the start of the rollover, copy the current log file to the archive path
        # and create a new log file with the following pattern:
        shutil.copy2(self.baseFilename, self.archive_path)
        self.baseFilename = os.path.join(os.path.dirname(self.baseFilename), self.name + self.file_extension) + time.strftime('-%Y-%m-%d-%H%M%S')
        dfn = self.rotation_filename(self.baseFilename)
        if os.path.exists(dfn):
            # Remove the destination file if it already exists
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)
        
        # if it hit the  the max_age_day Get the current date, Get the date of the last modification of the log file
        # then Calculate the age of the log file in days and remove the log file if it is older than the max age
        if self.max_age_days > 0:
            current_date = datetime.now().date()
            for file in os.listdir(self.log_path):
                if file.startswith(self.name):
                    file_path = os.path.join(self.log_path, file)
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
                    age_days = (current_date-file_date).days
                    if self.max_age_days <= age_days:
                        os.remove(file_path)

        # Determine the files that are in the log path but not in the archive path, copy that file from log/ path to archive_path
        # if dose not exist
        if self.archive_path:
            archive_file = list(set(os.listdir(self.log_path)) - set(os.listdir(self.archive_path)))
            for file in archive_file:
                file_path = os.path.join(self.log_path, file)
                shutil.copy2(file_path, self.archive_path)
                
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
                
        self.rolloverAt = self.computeRollover(currentTime)