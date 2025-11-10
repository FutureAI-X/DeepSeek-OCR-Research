def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t