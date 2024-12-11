import re


def text_splitter(text: str) -> list:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def check_yes_no(text: str) -> int:
    if "Yes" in text:
        return 1
    elif "No" in text:
        return 0
    else:
        return 0