import csv



def build_letter_ngrams_from_corpus(input_csv, output_index_file, max_len):
    with open(input_csv, 'r') as file:
        ll_gram = set()
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in csv_reader:
            ll_gram.update(extract_letter_grams(row[0], max_len))
        with open(output_index_file, 'w') as out_index:
            l = list(ll_gram)
            l.sort()
            out_index.write('#N number od letter grams\r')
            out_index.write(str(max_len) + '\r')
            out_index.write('#Index of Letter grams\r')
            for lg in l:
                out_index.write(lg)
                out_index.write('\r')
            print("Total  ", len(l), ' features generated')


def extract_letter_grams(main_str, max_len):
    str = '[' + main_str + ']'
    ret = set()
    for i in range(1, max_len + 1):
        for j in range(0, len(str) - i + 1):
            l_gram = str[j:(j + i)]
            if (l_gram.find(' ') == -1):
                ret.add(l_gram)
    return ret


def load_index(index_file):
    print('Loading the feature index')
    dict = {}
    max_len = 0
    with open(index_file, 'r') as file:
        line = file.readline().strip(' \r\n')
        index = 0

        while line:
            if line.startswith('#'):
                #skipp
                print('Line')
            elif max_len == 0:
                max_len = int(line)
            else:
                dict[line] = index
                index = index + 1
            line = file.readline().strip(' \r\n')
    model = {'index': dict, 'n': max_len}
    return model


def word_to_feature(word,  model):
    ret = [0] * len(model['index'])
    for feat in extract_letter_grams(word, model['n']):
        if feat in model['index']:
            ret[model['index'][feat]] = 1
    return ret


if __name__ == '__main__':
    build_letter_ngrams_from_corpus('./data/logit.csv', './data/lngrams.txt', 3)
    model = load_index('./data/lngrams.txt')
    print(word_to_feature('capable', model))
