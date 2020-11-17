"""
TODO:
1. Сделать декоратор для класса, который позволяет кэшировать некоторые внутренние методы,
используя кэш на диске.
"""

from contextlib import contextmanager
import sys
import os
import traceback
import json
import pickle
import logging
from .traceback_ import format_exception, format_traceback
import time

"""
TODO: add logging for all functions instead of print
"""

try:
    import requests
    from bs4 import BeautifulSoup as _BeautifulSoup
    import yaml
    import pandas as pd
except:
    imports = """
    import requests
    from bs4 import BeautifulSoup as _BeautifulSoup
    import yaml
    import pandas as pd
    """.split('\n')


    class TColors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[31m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


    for i in imports:
        try:
            exec(i.strip())
        except ModuleNotFoundError as e:
            print(TColors.WARNING + repr(e) + TColors.ENDC)


def head(df: pd.DataFrame, n=10):
    """
    pretty print dataframe
    """
    print(df.head(n).to_string())


# Used as @fs.cached
def cached(cache, key=lambda *args, **kwargs: args):
    """Decorator to wrap a function with a memoizing callable that saves
    results in a cache.

    main idea from cachetools.cached

    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            k = key(*args, **kwargs)
            try:
                return cache[k]
            except KeyError:
                pass  # key not found
            v = func(*args, **kwargs)
            try:
                cache[k] = v
            except ValueError:
                pass  # value too large
            return v

        return wrapper

    return decorator


@contextmanager
def cache_args_pkl(call_func, dumppath=''):
    # default_cache = cachetools.TTLCache(ttl=777600)   # ttl = 10 days
    # default_cache = cachetools.TTLCache(100, ttl=777600)   # ttl = 10 days

    # because tuple saved in dict by id in memory, and can have two identical keys but with different ids
    # cache = {tuple(k): v for k, v in self.read_dump(name, {}).items()}

    if dumppath and not dumppath.endswith('/'):
        dumppath = dumppath + '/'

    dumppath = dumppath + '__pkl_cashe_dump__/'
    os.makedirs(dumppath, exist_ok=True)

    dumpfile = dumppath + call_func.__name__ + '.pkl'

    try:
        with open(dumpfile, 'rb') as f:
            cache = pickle.load(f)
    except:
        cache = {}

    f = cached(cache=cache)(call_func)

    try:
        yield f

    except BaseException as e:  # example: KeyboardInterrupt
        write_file(dumpfile, cache, 'pkl')
        raise e

    write_file(dumpfile, cache, 'pkl')


def decor_dump_to_pkl(display_name, dumppath):
    """
    cache first hard_work function result to pkl file and load it instead work of it
    """

    # if not ISLOCAL or not DumpFlag or not os.path.exists(ISLOCAL):
    #
    #     def no_decorate(call_func):
    #         return call_func
    #
    #     return no_decorate

    # dumppath = dumppath + '/' + display_name + '/__pkl_cashe_dump__/'
    dumppath = dumppath + '/__pkl_cashe_dump__/'
    os.makedirs(dumppath, exist_ok=True)

    def decorator(call_func):
        def wrapper(*args, **kwargs):
            filename = dumppath + call_func.__name__ + '.pkl'
            if os.path.exists(filename):
                result = read_file(filename)
                return result
            else:
                result = call_func(*args, **kwargs)
                write_file(filename, result, 'pkl')
                return result

        return wrapper

    return decorator


def url_get(url, proxies=False, headers=None):
    if not proxies:
        proxies = {}
    with requests.Session() as session:
        return session.get(url, headers=headers, proxies=proxies).content


def soup(content='', headers=None):
    if content.startswith('http'):
        content = url_get(content, headers=headers)

    if content:
        _soup = _BeautifulSoup(content, 'lxml')
        # soup.find('title', attrs={'itemprop': "name"})
        return _soup
    return None


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, BaseException):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def jdumps(o, indent=2, ensure_ascii=False, sort_keys=True, cls=CustomJSONEncoder, **kwargs):
    return json.dumps(o, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, cls=cls, **kwargs)


def ydumps(o, sort_keys=True, **kwargs):
    # TODO: need read about custom encoders
    # allowed kwargs:
    # stream=None,
    # Dumper=Dumper,
    # default_style=None,
    # default_flow_style=False,
    # canonical=None,
    # indent=None,
    # width=None,
    # allow_unicode=None,
    # line_break=None,
    # encoding=None,
    # explicit_start=None,
    # explicit_end=None,
    # version=None,
    # tags=None,
    return yaml.safe_dump(o, allow_unicode=True, sort_keys=sort_keys, **kwargs)


def read_file(filename, filetype=True, errors='ignore', **kwargs):
    """
    read file with kwargs:
    Defult filetype = extpath of file

    filetype variants:
        pkl: pickle.load(f)
        yaml: yaml.load(f)
        json: json.load(f)
        None or another: f.read()

    additional args:
    mode = 'r' - mode kwarg of open func
        byte mode auto set encoding=None
    encoding = 'utf-8' - encoding kwarg of open func.

    """

    mode = kwargs.get('mode', 'r')
    encoding = kwargs.get('encoding', 'utf-8') if 'b' not in mode else None

    if filetype and isinstance(filetype, bool):
        filetype = os.path.splitext(filename)[-1][1:]
    try:
        if filetype == 'pkl':
            with open(filename, 'rb') as f:
                return pickle.load(f)

        with open(filename, mode=mode, encoding=encoding) as f:
            if filetype == 'yaml':
                return yaml.safe_load(f)
            elif filetype == "json":
                return json.loads(f.read())
            # elif filetype == 'csv':
            #     return list(csv.reader(f, **kwargs))
            else:
                return f.read()

    except Exception as e:
        logging.getLogger().error(''.join(format_exception(e)))
        if errors != 'ignore':
            raise e


def write_file(filename, obj, mode=None, sort_keys=False):
    try:
        ext = filename.rsplit('.')[-1]
        if mode is None and ext in {'json', 'yaml', 'pkl'}:
            raise ValueError(f'If you save file with .{ext} extesnion, you need to set mode')
            #
            # if ext in {'json', 'yml', 'pkl'}:
            #     print(f'WARNING!!! YOU SAVE {ext} IN RAW MODE')
            # with open(filename, 'w') as f:
            #     f.write(obj)

        elif mode == 'wb' or isinstance(obj, bytes):
            with open(filename, 'wb') as f:
                f.write(obj)

        elif mode == 'pkl':
            if hasattr(obj, 'to_pickle'):
                try:
                    return obj.to_pickle(filename)
                except Exception:
                    pass

            with open(filename, 'wb') as f:
                pickle.dump(obj, f)

        else:
            with open(filename, 'w') as f:
                if mode == 'json':
                    f.write(jdumps(obj, sort_keys=sort_keys))
                elif mode == 'yaml':
                    f.write(ydumps(obj, sort_keys=sort_keys))
                else:
                    f.write(obj)
        return True
    except Exception as e:
        logging.getLogger().error(''.join(format_exception(e)))
        return


@contextmanager
def catch_exceptions(*exceptions, message=None):
    """
    manager for catch exceptions
    Application examples:

    >>> with catch_exceptions():
    ...     1/0
    ZeroDivisionError('division by zero',)

    >>> with catch_exceptions(KeyError, ZeroDivisionError):
    ...     1/0
    ZeroDivisionError('division by zero',)

    >>> with catch_exceptions(KeyError):
    ...     1/0
    ...
    Traceback (most recent call last):
          ...
    ZeroDivisionError: division by zero
    """
    if not exceptions:
        exceptions = (Exception,)
    try:
        yield
    except exceptions as e:
        try:
            if message is None:
                message = 'MANAGER CATCH '

            if message:
                print(message + f'Exception: {repr(e)} \n traceback: {traceback.format_exc()}')
        except Exception as e:
            pass

    return True


def logged(*exceptions, logger=None, level='ERROR'):
    if not exceptions:
        exceptions = (Exception,)

    if logger is None:
        logger = logging.getLogger()

    print_log = getattr(logger, level.lower())

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except exceptions as e:
                print_log(f"{repr(e)} in {func.__name__}, \n"
                          f"{format_traceback(sys.exc_info()[2].tb_next)}")
                raise e

        return wrapper

    return decorator


@contextmanager
def log_errors(logger=None, *exceptions, level='ERROR'):
    if not exceptions:
        exceptions = (Exception,)

    if logger is None:
        logger = logging.getLogger()

    logger = logger.getChild('LOG_ERRORS')

    print_log = getattr(logger, level.lower())

    try:
        yield
    except exceptions as e:
        print_log(''.join(format_exception(e)))
        raise e


class Timer:
    asctime = time.asctime
    logger = logging.getLogger('Timer')

    def __init__(self, name=None, logger=None, level='DEBUG', alert_period=10):

        if logger is not None:
            self.logger = logger

        if name is not None:
            self.logger = self.logger.getChild(name)

        self.alert_period = alert_period
        self.t0 = time.time()
        self.next_alert_time = self.t0 + self.alert_period
        self.data_quantity = None
        self.print = getattr(self.logger, level.lower())
        self.counter = 0

    def start(self, data_quantity=None, message=''):
        self.t0 = time.time()
        self.counter = 0
        self.data_quantity = data_quantity
        if message:
            self.print(message)

    def time(self, isprint=False):
        if isprint:
            self.print(f'Elapsed time {time.time() - self.t0:.3f} s, ')
        return time.time()

    def reset(self):
        self.t0 = time.time()

    def count(self, message=''):
        self.counter += 1
        self.checktime(self.counter, message=message)

    def checktime(self, N=0, message=''):
        if time.time() > self.next_alert_time:
            dt = time.time() - self.t0

            if N and self.data_quantity:
                eta = dt * (self.data_quantity - N) / (N + 1)
                self.print(
                    f'Processed {N}/{self.data_quantity} rows,'
                    f'elapsed {dt:.2f} s,'
                    f'remaining time {eta / 60:.1f} min, remaining ratio {100 * N / self.data_quantity:.1f} %' + message
                )
            else:
                self.print(
                    f'Elapsed time {dt:.2f} s, ' + message
                )
            self.next_alert_time = time.time() + self.alert_period

    def iter(self, iterator, Q=None):
        """
        Example usage:
        >>> l = [1,2,3,4]
        >>> for i in self.iter(l):
        ...     time.sleep(5)
        """

        self.t0 = time.time()

        if Q is not None:
            self.data_quantity = Q
            self.print(f'Start iterate through {self.data_quantity} items')
            f_str = f' from {self.data_quantity}'
        else:
            try:
                self.data_quantity = len(iterator)
                self.print(f'Start iterate through {self.data_quantity} items')
                f_str = f' from {self.data_quantity}'
            except Exception:
                self.data_quantity = None
                f_str = ''

        for N, args in enumerate(iterator):
            self.checktime(N, message=f'read {N} values' + f_str)
            yield args

    def decor(self, call_func, Q=None):
        """
        decorate function for Dataframe.apply.
        For example:
        >>> l = ['1', '2', '3', '4']
        ... f = self.decor(int, len(l))
        ... for i in l:
        ...     print(f(i))
        ...     time.sleep(5)
        """

        self.t0 = time.time()

        if Q is not None:
            self.data_quantity = Q
            self.print(f'Start iterate through {self.data_quantity} items')
            f_str = f' from {self.data_quantity}'
        else:
            self.data_quantity = 1
            f_str = ''

        self_counter = 0

        def wrapper(*args, **kwargs):
            nonlocal self_counter
            self.checktime(self_counter, message=f'execute call #{self_counter}' + f_str)
            self_counter += 1
            return call_func(*args, **kwargs)

        return wrapper

    @contextmanager
    def manager(self):
        self.start('start timer')
        yield
        self.time()
        return True
