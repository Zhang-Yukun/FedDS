from functools import wraps
from typing import Any
import atexit as _atexit
import sys as _sys

from loguru._logger import Logger as _Logger
from loguru._logger import Core as _Core
from loguru import _defaults



class Logger(object):
    def __init__(self):
        self._logger = _Logger(
            core=_Core(),
            exception=None,
            depth=1,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patchers=[],
            extra={},
        )
        if _defaults.LOGURU_AUTOINIT and _sys.stderr:
            self._logger.add(_sys.stderr)

        _atexit.register(self._logger.remove)

    def add(
            self,
            sink,
            *,
            level="DEBUG",
            fmt="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            flt=None,
            colorize=None,
            serialize=False,
            backtrace=True,
            diagnose=True,
            enqueue=False,
            context=None,
            catch=True,
            **kwargs
    ) -> int:
        return self._logger.add(
            sink,
            level=level,
            format=fmt,
            filter=flt,
            colorize=colorize,
            serialize=serialize,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=enqueue,
            context=context,
            catch=catch,
            **kwargs
        )

    def remove(self, handler_id: int | None = None):
        self._logger.remove(handler_id)

    def complete(self):
        self._logger.complete()

    def trace(self, obj: Any):
        if not callable(obj):
            self._logger.trace(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.trace(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.trace(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def debug(self, obj: Any):
        if not callable(obj):
            self._logger.debug(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.debug(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.debug(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def info(self, obj: Any):
        if not callable(obj):
            self._logger.info(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.info(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.info(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def success(self, obj: Any):
        if not callable(obj):
            self._logger.success(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.success(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.success(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def warning(self, obj: Any):
        if not callable(obj):
            self._logger.warning(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.warning(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.warning(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def error(self, obj: Any):
        if not callable(obj):
            self._logger.error(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.error(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.error(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def critical(self, obj: Any):
        if not callable(obj):
            self._logger.critical(obj)
        else:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self._logger.critical(f"Calling function {obj.__name__}")
                result = obj(*args, **kwargs)
                self._logger.critical(f"Function {obj.__name__} returned")
                return result
            return wrapper

    def log(self, level: str, message: str | None = None):
        if message is not None:
            self._logger.log(level, message)
        else:
            def decorator(obj: Any):
                @wraps(obj)
                def wrapper(*args, **kwargs):
                    self._logger.log(level, f"Calling function {obj.__name__}")
                    result = obj(*args, **kwargs)
                    self._logger.log(level, f"Function {obj.__name__} returned")
                    return result
                return wrapper
            return decorator


logger = Logger()
