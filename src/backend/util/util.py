import datetime


def timestamp_string():
    return datetime.datetime.now(datetime.UTC).isoformat()


def split_lines(ai_response):
    return ai_response.strip().split("\n")
