import logging

def setup_logger(name=__name__, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to prevent duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set the level
    logger.setLevel(level)
    
    # Create a console handler and set its level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create a formatter that includes timestamp, level, filename, line number, and message
    formatter = logging.Formatter(
        # '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        '[%(levelname)s]: %(message)s - %(filename)s:%(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set the formatter for the handler
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (another common cause of duplication)
    logger.propagate = False
    
    return logger

# # Example usage
# logger = setup_logger()

# def some_function():
#     logger.info("This is an info message")
#     logger.warning("This is a warning message")
#     logger.error("This is an error message")

# if __name__ == "__main__":
#     some_function()