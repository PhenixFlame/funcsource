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

    m = max(map(len, code[start:end]))
    x = s.lstrip(' \t')
    l = len(s.expandtabs()) - len(x) - 2
    s = '-' * l, '>|', x.strip(' \n\t'), '|<'
    s = ''.join(s).ljust(m, '-') + '\n'
    yield s
    # --END--->|code[line]|<------------

    yield from code[line + 1: end]


def format_exception(e, need_locals=True):
    """
    format message about exception e with local variables
    :param e:
    :param need_locals:
    :return:
    """
    tb = sys.exc_info()[2]
    exception = "{}: {}\n".format(type(e).__name__, str(e))
    l_tb = list(tb_gen(tb))

    ss = [exception]
    for tb_i in l_tb[:-1]:
        ss.append('  ' + frame_format(tb_i))
        for c in code_window(tb_i, 3):
            ss.append('    ' + c)

    last = l_tb[-1]
    ss.append('  ' + frame_format(last))
    for c in code_window(last, None):
        ss.append('    ' + c)

    if need_locals:
        ss.append('LOCALS:\n')
        for l in locals_gen(last):
            ss.append('    ' + l)
    return ss
