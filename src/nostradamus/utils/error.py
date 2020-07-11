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


def check_is_in(arg, list_of_values):
    if arg in list_of_values:
        pass
    else:
        raise ValueError("Wrong argument value")

def check_key_is_in(list_of_keys, arg):
    if set(list_of_keys).issubset(arg):
        pass
    else:
        raise ValueError("Wrong argument value")