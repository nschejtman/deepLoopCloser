class MathUtils:
    @staticmethod
    def compressed_size(value: int, compression: float):
        return int(round(value * ((100 - compression) / 100)))
