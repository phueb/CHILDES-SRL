from typing import List


def make_srl_string(words: List[str],
                    tags: List[str]) -> str:
    frame = []
    chunk = []

    for (token, tag) in zip(words, tags):
        if tag.startswith("I-"):
            chunk.append(token)
        else:
            if chunk:
                frame.append("[" + " ".join(chunk) + "]")
                chunk = []

            if tag.startswith("B-"):
                chunk.append(tag[2:] + ": " + token)
            elif tag == "O":
                frame.append(token)

    if chunk:
        frame.append("[" + " ".join(chunk) + "]")

    return " ".join(frame)