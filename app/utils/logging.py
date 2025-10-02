import os, sys, time, uuid, json, logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON = os.getenv("LOG_JSON", "1") not in ("0", "false", "False")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            data.update(extra)
        return json.dumps(data, ensure_ascii=False)


def get_logger(name="app"):
    logger = logging.getLogger(name)
    if logger.handlers:  # prevent double handlers
        return logger
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if LOG_JSON
        else logging.Formatter("[%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def new_request_id():
    return uuid.uuid4().hex[:12]


def time_block():
    t0 = time.time()
    return lambda: round(time.time() - t0, 3)
