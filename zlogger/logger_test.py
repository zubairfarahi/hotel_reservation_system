from logger import ZLogger
import configparser
import unittest
import threading
import re
import os
import time

path_file = "./config/logging.ini"
config = configparser.ConfigParser()
config.read(path_file)



class LoggerThreadTest(unittest.TestCase):
    def setUp(self):
        self.logger = ZLogger("hotel_reservation_system", config)
        self.thread_one_count = {"23432-324-2343223423-573532": 0}
        self.thread_two_count = {"573532-23432-324-2343223423-324234": 0}

    def thread_one(self):
        for _ in range(14):
            self.logger.with_request_id("23432-324-2343223423-573532").debug(str("fill not processed"))

    def thread_two(self):
        for _ in range(4):
            self.logger.with_request_id("573532-23432-324-2343223423-324234").debug(str("fill not processed"))

    def test_logger_threads(self):
        # Create the first thread
        t1 = threading.Thread(target=self.thread_one)

        # Create the second thread
        t2 = threading.Thread(target=self.thread_two)

        # Start both threads
        t1.start()
        t2.start()

        # Wait for both threads to finish
        t1.join()
        t2.join()

        log_file = sorted(os.listdir("../logs"))[-1]
        log_file = os.path.join("../logs", log_file)

        with open(log_file, "r") as file:
            for line in file:
                match = re.search(r"requestID: ([\w-]+)", line)
                if match:
                    request_id = match.group(1)
                    if request_id == "23432-324-2343223423-573532":
                        self.thread_one_count["23432-324-2343223423-573532"] += 1
                    elif request_id == "573532-23432-324-2343223423-324234":
                        self.thread_two_count["573532-23432-324-2343223423-324234"] += 1

        self.assertEqual(self.thread_one_count["23432-324-2343223423-573532"], 14, "Thread one count mismatch")
        self.assertEqual(self.thread_two_count["573532-23432-324-2343223423-324234"], 4, "Thread two count mismatch")
        


def custom_test():
    logger = ZLogger("hotel_reservation_system", config)
    for _ in range(3):
        logger.with_request_id("573532-23432-324-2343223423-324234").debug(str("fill not processed"))
        logger.with_module_name("test demo").with_request_id("1234").with_additional_data({
                "time": "4s",
                'uuid': 12345
            }).info('done')
            
        logger.with_additional_data({
            "module_name": "test desc",
            "UUID": "2423-3544-3s3d-gr32"
        }).exception(str("file not found"))
        
        
        logger.with_additional_data({
            "request_id": "573532-23432-324-2343223423-324234",
            "module_name": "test desc",
            "timing": "3 ms",
            "UUID": "2423-3544-3s3d-gr32"
        }).info(str("file not found"))
        time.sleep(1)
        logger.with_request_id("23432-324-324234").info(str("processing file"))
        logger.with_additional_data({
            "request_id": "23432-324-324234",
            "module_name": "error testing",
            "path": "/home/user/file.txt",
            "UUID": "1111-3544-3s3d-gr32"
        }).error(str("file is corrupted"))
        time.sleep(1)
        logger.with_request_id("12345").success(str("file successfully processed"))
        logger.with_additional_data({
            "request_id": "12345",
            "module_name": "test desc",
            "timing": "3 ms"
        }).info(str("fill successfully processed"))
        time.sleep(1)
        logger.with_request_id("573532-23432-324-2343223423-324234").warning(str("can't process file"))
        logger.with_additional_data({
            "request_id": "573532-23432-324-2343223423-324234",
            "module_name": "test desc",
            "timing": "3 ms"
        }).reject(str("file was rejected"))
        time.sleep(1)
        logger.with_additional_data({
            "request_id": "1234_av123_1234",
            "module_name": "route save_file",
            "timing": "3 ms"
        }).fatal(str("file is fatal"))
        time.sleep(1)
        logger.warning("Hello World")
        time.sleep(1)
        logger.reject("Hello World")
        time.sleep(1)
        logger.success("done!")
        time.sleep(1)
        logger.with_additional_data({
            "request_id": "23432-324-324234",
            "module_name": "error testing",
            "path": "/home/user/file.txt",
            "UUID": "1111-3544-3s3d-gr32"
        }).error(str("file is corrupted"))
        time.sleep(1)
        logger.with_request_id("23432-324-324234").info(str("processing file"))
        time.sleep(1)
        logger.with_additional_data({
            "request_id": "23432-324-324234",
            "module_name": "error testing",
            "path": "/home/user/file.txt",
            "UUID": "1111-3544-3s3d-gr32"
        }).error(str("file is corrupted"))
        time.sleep(1)
        logger.with_request_id("23432-324-324234").info(str("processing file"))
        logger.with_additional_data({
            "request_id": "23432-324-324234",
            "module_name": "error testing",
            "path": "/home/user/file.txt",
            "UUID": "1111-3544-3s3d-gr32"
        }).error(str("file is corrupted"))
        
        try:
            x = 1/0
        except Exception as e:
            logger.with_request_id("23432-324-324234").exception(e)

if __name__ == '__main__':
    # if you want to test without threads, uncomment the custom_test() function and comment unittest.main()
    # unittest.main()
    custom_test()