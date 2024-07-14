from typing import List

PAD_TOKEN = (0, 0)
START_TOKEN = (1, 0)
END_TOKEN = (2, 0)

def pad_spectrum(spectrum: List[tuple[int, int]], max_len: int):
    s = [START_TOKEN] + spectrum + [END_TOKEN]
    y_offset = max_len - len(s)
    if y_offset:
        s += y_offset*[PAD_TOKEN]
    return s

