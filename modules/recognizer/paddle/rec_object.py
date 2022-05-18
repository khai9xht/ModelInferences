class RecObject:
    def __init__(self, text: str, score: float):
        self.text = text
        self.score = score

    def __str__(self) -> str:
        return "'{}' ({:.3f})".format(self.text, self.score)