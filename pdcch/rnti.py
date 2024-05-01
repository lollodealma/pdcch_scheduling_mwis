

class RntiGenerator:
    def __init__(self):
        self.p0 = 320  # skip 300 first C-RNTIs values in range of RA-RNTI - 61 + 7*37 = 320
        self.n_rnti = 0
        self.coef = 37
        self.p = None

    def generate_rnti(self):
        if self.n_rnti == 0:
            rnti = self.p0
        else:
            rnti = (self.p + self.coef) % 65536

        self.p = rnti
        self.n_rnti += 1

        return rnti

