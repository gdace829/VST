import logging
import sys
from typing import Any


class _LoggerShim:
    def __init__(self):
        self._logger = logging.getLogger("lmms_eval")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self._logger.addHandler(handler)

    def remove(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)

    def add(self, sink, level="INFO", format=None, colorize=False, **kwargs: Any) -> None:
        del colorize, format, kwargs
        handler = logging.StreamHandler(sink)
        handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._logger.addHandler(handler)

    def bind(self, **kwargs: Any):
        del kwargs
        return self

    def opt(self, **kwargs: Any):
        del kwargs
        return self

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **kwargs)


logger = _LoggerShim()
