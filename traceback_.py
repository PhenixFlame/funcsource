import sys
import inspect


def join(*args, s=''):
    return s.join(args)


def tb_gen(tb):
    while tb:
        yield tb
        tb = tb.tb_next


def locals_gen(tb):
    for name, obj in tb.tb_frame.f_locals.items():
        yield join(name, ' -> ', str(obj), '\n')


def frame_format(tb):
    code = tb.tb_frame.f_code
    filename = code.co_filename
    lineno = tb.tb_lineno
    name = code.co_name
    frame_string = f'File "{filename}", line {lineno}, in {name}\n'
    return frame_string


def code_window(tb, window=None):
    code, start_line = inspect.getsourcelines(tb)
    line = tb.tb_lineno - start_line

    if window is None:
        start = 0
        end = None
    else:
        w = window // 2
        start = line - w
        end = line + w + 1

    yield from code[start:line]

    # ----->> code[line] <------------
    s = code[line]

    x = s.lstrip(' \t')
    l = len(s.expandtabs()) - len(x) - 3
    s = join('>' * l, '>| ', x)
    yield s
    # --END--->|code[line]|<------------

    yield from code[line + 1: end]


def format_traceback(tb, need_locals=True):
    while tb.tb_next:
        yield '  ' + frame_format(tb)
        for c in code_window(tb, 3):
            yield '    ' + c
        tb = tb.tb_next

    yield '  ' + frame_format(tb)
    for c in code_window(tb, None):
        yield '    ' + c

    if need_locals:
        yield 'LOCALS:\n'
        for l in locals_gen(tb):
            yield '    ' + l


def format_exception(e, need_locals=True):
    """

    Generator

    format message about exception e with local variables. For Example:

       File "/HOME/PyCharmProjects/DuckHunterBot/funcsource/__init__.py", line 348, in log_errors
         try:
     ------>|yield|<-------------
         except exceptions as e:

    :param e: exception object
    :param need_locals: flag

    """
    tb = sys.exc_info()[2]

    # ExampleError: example object has no attribute 'getChild'
    yield "{}: {}\n".format(type(e).__name__, str(e))

    yield from format_traceback(tb, need_locals)
    yield "{}: {}\n".format(type(e).__name__, str(e))  # duplicate for convenience
