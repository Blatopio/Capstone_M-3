from datetime import datetime

def get_current_date():
    """
    Get the current date.
    args: 
        None
    returns: 
        str: with the current date in the format "YYYY-MM-DD HH:MM:SS"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")