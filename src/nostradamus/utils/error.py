'''
Error management
'''


def exception_type(arg, typed):
    if isinstance(arg, typed):
        pass
    else:
        raise Exception("Wrong Type")
    pass
