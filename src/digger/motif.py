# A class to manage and use motif matrices

import csv
import json
import numpy as np
from receptor_utils import simple_bio_seq as simple


class Motif:
    def __init__(self, name, seqs=None, stream=None):
        self.name = name
        self.matrix = []
        self.consensus = None
        self.conserved_consensus = None
        self.consensus_prob = None
        self.likelihood_threshold = None
        self.logo = None
        # logo check per window sequence: the leader scan checks the same windows again for every BLAST hit near a gene, so
        # check each once. Cleared whenever the matrix is recalculated (calc_update)
        self.logo_cache = {}

        if seqs is not None:
            self.calc_prob_matrix(seqs)
        elif stream is not None:
            self.read_prob_matrix(stream)

    def calc_update(self):
        self.logo_cache = {}
        # for window_likelihoods: per motif position, the factor calc_prob uses for each byte (1.0 where calc_prob skips the base)
        self.base_tables = []
        for m in self.matrix:
            table = np.ones(256)
            for b, p in m.items():
                table[ord(b)] = p
            self.base_tables.append(table)
        self.calc_consensus()
        self.consensus_prob = self.calc_prob(self.consensus)
        self.calc_conserved_consensus()

    def calc_prob_matrix(self, seqs):
        al = zip(*seqs)

        self.matrix = []
        for z in al:
            pseudo = min(len(seqs)/10, 1)
            dist = {'A': pseudo, 'C': pseudo, 'G': pseudo, 'T': pseudo}  # inc. pseudocount
            total = 4*pseudo
            for base in z:
                if base in ['A', 'C', 'G', 'T']:
                    dist[base] += 1.0
                    total += 1
            for base in ['A', 'C', 'G', 'T']:
                dist[base] = dist[base]/total
            dist_sum = 0
            for base in ['A', 'C', 'G', 'T']:
                dist_sum += dist[base]
            if dist_sum < 0.99 or dist_sum > 1.01:
                print('error in pcm calc')
            self.matrix.append(dist)
        self.calc_update()

    def write_prob_matrix(self, fo):
        fo.write('%.6E\n' % self.likelihood_threshold)
        if self.logo:
            fo.write(f'Logo: {json.dumps(self.logo)}\n')

        writer = csv.writer(fo)
        writer.writerow(['A', 'C', 'G', 'T'])
        for row in self.matrix:
            writer.writerow([round(row[base], 4) for base in ['A', 'C', 'G', 'T']])

    def read_prob_matrix(self, fi):
        self.likelihood_threshold = float(fi.readline().replace('\n', ''))
        pos = fi.tell()
        logo = fi.readline().replace('\n', '')

        if logo and 'Logo: ' in logo:
            self.logo = json.loads(logo.split('Logo: ')[1])
        else:
            fi.seek(pos)

        reader = csv.DictReader(fi)
        self.matrix = list(reader)
        for row in self.matrix:
            for k,v in row.items():
                row[k] = float(v)
        self.calc_update()

    def calc_consensus(self):
        self.consensus = ''
        for row in self.matrix:
            c_pos = None
            c_prob = 0.0
            for base in ['A', 'C', 'G', 'T']:
                if row[base] > c_prob:
                    c_prob = row[base]
                    c_pos = base
            self.consensus += c_pos if c_pos is not None else '-'

    def calc_conserved_consensus(self):
        self.conserved_consensus = ''
        for row in self.matrix:
            c_pos = None
            for base in ['A', 'C', 'G', 'T']:
                if row[base] >= 0.75:
                    c_pos = base
                    break
            self.conserved_consensus += c_pos if c_pos is not None else '-'

    def calc_prob(self, seq):
        prob = 1.0
        for b, m in zip(list(seq),self.matrix):
            if b in m:
                prob = prob * m[b]
        return prob

    def calc_likelihood(self, seq):
        return self.calc_prob(seq) / self.consensus_prob


    # calc_likelihood of every window of the motif's length in seq, as a numpy array. Each window is multiplied base by base in the
    # same order as calc_prob, in the same float64 arithmetic, so the values are bit-identical to calc_likelihood
    def window_likelihoods(self, seq):
        codes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        n = len(codes) - len(self.base_tables) + 1
        prob = np.ones(n)
        for j, table in enumerate(self.base_tables):
            prob = prob * table[codes[j:j + n]]
        return prob / self.consensus_prob

    def find_non_conserved(self, seq):
        non_conserved = 0
        res = ''
        for s, ref in zip(list(seq), list(self.conserved_consensus)):
            if ref != '-' and s != ref:
                non_conserved += 1
                res += s
            else:
                res += '-'

        return non_conserved, res

    def check_logo(self, seq):
        if seq not in self.logo_cache:
            self.logo_cache[seq] = self.check_logo_uncached(seq)
        return self.logo_cache[seq]

    def check_logo_uncached(self, seq):
        if self.logo is None:
            return True
        
        seq = simple.translate(seq)
        
        if len(seq) != len(self.logo):
            return None

        ex_count = 0
        for i, base in enumerate(seq):
            if base not in self.logo[i]:
                ex_count += 1

        #print(ex_count)

        #if ex_count == 5:
        #    breakpoint()

        return ex_count < 3
