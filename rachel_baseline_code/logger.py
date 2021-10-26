import logging

def make_logger(name=None):
    #1 logger instance를 만든다.
    logger = logging.getLogger(name)

    #2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    #3 formatter 지정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    #4 handler instance 생성
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename="performance.log")
    
    #5 handler 별로 다른 level 설정
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    #6 handler 출력 format 지정
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    #7 logger에 handler 추가
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger