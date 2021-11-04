from collections import Counter
from functools import reduce

import numpy as np
from nltk.translate.bleu_score import ngrams


class metric_names:
    jaccard = "Jaccard"
    sorensen = "Sorensen"
    canberra = "Canberra"
    minkowski = "Minkowski"
    BLEU = "BLEU"
    SelfBLEU = "SelfBLEU"


def get_ngrams(sentences, n):
    f = lambda x: (list(ngrams(x, n)) if len(x) >= n else [])
    return list(map(f, sentences))


class MultisetDistances:

    def __init__(self, references, min_n=3, max_n=5):
        super().__init__()
        # print('multiset distances init upto {}!'.format(max_n))
        self.references = references
        self.max_n = max_n
        self.min_n = min_n
        assert self.max_n >= self.min_n
        assert self.min_n >= 1
        self.ref_ngrams = self._get_ngrams(references)

    def get_cached_fields(self):
        return self.ref_ngrams,

    def _get_ngrams(self, sentences):
        samples_size = len(sentences)
        all_counters = [Counter([x for y in get_ngrams(sentences, n + 1) for x in y])
                        for n in range(self.max_n)]
        for n_counter in all_counters:
            for k in n_counter.keys():
                n_counter[k] /= samples_size
        return all_counters

    def get_ngram_stuff(self, sentences):
        sample_ngrams = self._get_ngrams(sentences)
        ngrams_intersection = [sample_ngrams[i] & self.ref_ngrams[i]
                               for i in range(self.max_n)]  # intersection:  min(c[x], d[x])
        ngrams_union = [sample_ngrams[i] | self.ref_ngrams[i]
                        for i in range(self.max_n)]  # union:  max(c[x], d[x])
        ngrams_abs_diff = [ngrams_union[i] - ngrams_intersection[i] \
                           for i in range(self.max_n)]
        ngrams_added = [sample_ngrams[i] + self.ref_ngrams[i]
                        for i in range(self.max_n)]

        return ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added

    def _final_average(self, score_value):
        return np.power(reduce(lambda x, y: x * y, score_value), 1. / float(len(score_value)))

    def _jaccard(self, ngrams_intersection, ngrams_union):
        jaccard_value = [float(sum(ngrams_intersection[n].values())) / sum(ngrams_union[n].values()) for n in
                         range(self.max_n)]
        return jaccard_value

    def get_jaccard_score(self, sentences):
        print('Jaccard distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = self.get_ngram_stuff(sentences)

        jaccard_value = self._jaccard(ngrams_intersection=ngrams_intersection, ngrams_union=ngrams_union)

        return {n: self._final_average(jaccard_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def _sorensen(self, ngrams_abs_diff, ngrams_added):
        sorensen_value = [float(sum(ngrams_abs_diff[n].values())) / sum(ngrams_added[n].values()) for n in
                          range(self.max_n)]
        return sorensen_value

    def get_sorensen_score(self, sentences):
        print('Sorensen distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = self.get_ngram_stuff(sentences)

        sorensen_value = self._sorensen(ngrams_abs_diff=ngrams_abs_diff, ngrams_added=ngrams_added)

        return {n: self._final_average(sorensen_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def _canberra(self, ngrams_abs_diff, ngrams_added):
        canberra_value = [np.sum([ngrams_abs_diff[n][key] / float(ngrams_added[n][key]) for key in ngrams_abs_diff[n]])
                          for n in range(self.max_n)]
        return canberra_value

    def get_canberra_score(self, sentences):
        print('Canberra distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = self.get_ngram_stuff(sentences)
        canberra_value = self._canberra(ngrams_abs_diff=ngrams_abs_diff, ngrams_added=ngrams_added)
        return {n: self._final_average(canberra_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def _minkowski(self, ngrams_abs_diff, p):
        minkowski_value = [np.power(np.sum(np.power(list(ngrams_abs_diff[n].values()), p)), 1. / p) for n in
                           range(self.max_n)]
        return minkowski_value

    def get_minkowski_score(self, sentences, p):
        print('Minkowski (p={}) distances preprocess upto {}!'.format(p, self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = self.get_ngram_stuff(sentences)

        minkowski_value = self._minkowski(ngrams_abs_diff=ngrams_abs_diff, p=p)

        return {n: self._final_average(minkowski_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def get_all_score(self, sentences, max_mikowski_order=3):
        print('multiset distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = self.get_ngram_stuff(sentences)

        temp_results = {}

        print('multiset distances evaluating upto {}!'.format(self.max_n))
        temp_results[metric_names.jaccard] = self._jaccard(ngrams_intersection=ngrams_intersection,
                                                           ngrams_union=ngrams_union)
        temp_results[metric_names.sorensen] = self._sorensen(ngrams_abs_diff=ngrams_abs_diff, ngrams_added=ngrams_added)
        temp_results[metric_names.canberra] = self._canberra(ngrams_abs_diff=ngrams_abs_diff, ngrams_added=ngrams_added)
        for p in range(1, max_mikowski_order + 1):
            temp_results['p%d-%s' % (p, metric_names.minkowski)] = self._minkowski(ngrams_abs_diff=ngrams_abs_diff, p=p)

        result = {}
        for key in temp_results:
            for n in range(self.min_n, self.max_n + 1):
                result[key + '%d' % n] = self._final_average(temp_results[key][:n])
        return result


if __name__ == "__main__":
    # print(MultisetDistances([['saadi', 'saadi']], max_n=2).get_all_score([['saadi', 'ferdowsi']]))
    # print(MultisetDistances([['saadi', 'saadi']], max_n=2).get_all_score([['saadi', 'saadi']]))

    ref1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever',
            'heed', 'Party', 'commands']
    ref2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always',
            'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    ref3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions',
            'of', 'the', 'party']
    sen1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys',
            'the', 'commands', 'of', 'the', 'party']
    sen2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 'in', 'world', 'history']

    references = [ref1, ref2, ref3]
    sentences = [sen1, sen2]

    msd = MultisetDistances(references=references)
    print(msd.get_jaccard_score(sentences=sentences))
