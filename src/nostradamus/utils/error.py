'''
Error management
'''


def exception_type(arg, typed) -> None:
    if isinstance(arg, typed):
        pass
    else:
        raise Exception("Wrong Type")
    pass

def check_is_int(arg) -> None:

    def is_integer(n) -> bool:
        return n % 1 == 0

    if is_integer(arg):
        pass
    else:
        raise ValueError("Only integers are allowed")
    pass

