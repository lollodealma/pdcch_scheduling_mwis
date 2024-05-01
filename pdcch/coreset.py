
from pdcch.utils import maskCce


class Coreset:
    # CORESET is the set of CCEs that can be allocated for PDCCH
    def __init__(self, n_cce, id):
        self.id = id  # ID
        self.n_cce = n_cce  # total number of CCEs
        self.allocatedCce = 0  # all 0's in binary - nothing allocated

    def check_cce_is_free(self, startCce, n_al):
        # checks if it is possible to allocate n_al CCEs starting from startCce
        cceToCheck = maskCce(startCce, n_al)
        # check if there is a conflict with already allocated CCe's
        alloc_conflicts = self.allocatedCce & cceToCheck
        return alloc_conflicts == 0

    def allocate_cce(self, startCce, n_al):
        if self.check_cce_is_free(startCce, n_al):
            # print("successful allocation :",n_al, " CCEs starting from ",startCce, " in ", hex(self.allocatedCce))
            # allocate n_al Cce's from the startCce+1-th position
            self.allocatedCce = self.allocatedCce | maskCce(startCce, n_al)
            success = True
        else:
            success = False
            # print("unsuccessful allocation :",n_al, " CCEs starting from ",startCce, " in ", hex(self.allocatedCce))
        return success

    def free_up_cce(self, startCce, n_al):
        # free up the n_al Cce's starting from startCce
        allCceMask = (1 << (self.n_cce + 1)) - 1
        self.allocatedCce = self.allocatedCce & (allCceMask - maskCce(startCce, n_al))

    def reset_cce(self):
        self.allocatedCce = 0


# class SearchSpace:
#     def __init__(self, id, coreset, n_pdcch_candidates_vs_al):
#         self.id = id
#         self.coreset = coreset
#         self.n_pdcch_candidates_vs_al = n_pdcch_candidates_vs_al  # number of possible candidates for each AL
#
#     def find_start_cce_candidates(self, nsf, n_rnti, n_al):
#         # computes candidates for the start of CCEs for a given AL and RNTI
#         # nsf -> slot number
#         # n_rnti -> RNTI identifying [user (optional), DCI type]
#         # n_al = Aggregation Level in [1,2,4,8,16]
#         n_ci = 0
#         n_pdcch_candidates = self.n_pdcch_candidates_vs_al[al_to_al_idx(n_al)]
#         output = arr.array('I', [])
#         for ii in range(n_pdcch_candidates):
#             # hashing function
#             rand_alloc = (Yp(self.coreset.id, nsf, n_rnti) + (ii * self.coreset.n_cce) //
#                           (n_al * n_pdcch_candidates) + n_ci)
#             output.insert(ii, n_al * (rand_alloc % (self.coreset.n_cce // n_al)))
#         return output

