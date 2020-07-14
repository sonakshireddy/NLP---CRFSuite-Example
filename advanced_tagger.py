import os, re,collections, sys
from hw2_corpus_tool import get_data
import pycrfsuite as crf

SPEAKER = 'speaker'
ACT_TAG = 'act_tag'
BOD = 'BEGINNING_OF_DIALOG'
POS_BIGRAM_DICT = {}
TOKEN_BIGRAM_DICT = {}
set_pos = set()


def create_pos_tag_list(pos_list, tag):
    result = []
    if not pos_list or (len(pos_list) == 1 and pos_list[0][0] == '.' and pos_list[0][1] == '.'):
        return ['NO_WORDS']
    for i, tup in enumerate(pos_list):
        pos = parse_pos(tup[1])
        result.append(tag + 'POS_' + pos)
        result.append(tag + 'TOKEN_' + tup[0])
        if i < len(pos_list) - 1:
            result.append(tag +  parse_pos(tup[1]) + "_" + parse_pos(pos_list[i + 1][1]))
            result.append(tag +  tup[0] + "_" + pos_list[i + 1][0])
    return result


def create_features(dialog):
    dialog_features = []
    labels = []
    prev_speak = None
    prev_pos = None
    for i, utterance in enumerate(dialog):
        features = []
        if ACT_TAG in utterance._fields:
            act = getattr(utterance, ACT_TAG)
            labels.append(act)
        postag_list = getattr(utterance, 'pos')
        result = create_pos_tag_list(postag_list, '')
        if postag_list and not (len(postag_list) == 1 and postag_list[0][0] == '.' and postag_list[0][1] == '.') and len(postag_list)<5:
            features.append('POS_LENGTH_'+str(len(postag_list)))
        if prev_pos:
            prev_result = create_pos_tag_list(prev_pos,'PREV_')
            result.extend(prev_result)
        prev_pos = postag_list
        if i > 0:
            if not prev_speak == getattr(utterance, SPEAKER):
                features.append('SPEAKER_CHANGE')
            else:
                features.append("SPEAKER_UNCHANGED")
        else:
            features.append(BOD)
        features.extend(result)
        prev_speak = getattr(utterance, SPEAKER)
        dialog_features.append(features)
    return dialog_features, labels


def create_trainer(X, Y):
    trainer = crf.Trainer()
    for x, y in zip(X, Y):
        trainer.append(x, y)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    return trainer


def evaulate_tagger(data, outputfile):
    tagger = crf.Tagger()
    tagger.open('advanced.crfsuite')
    total = 0
    pred_correct = 0
    f = open(outputfile, 'w')
    for dialog in data:
        x, y = create_features(dialog)
        pred = (tagger.tag(x))
        for p in pred:
            f.write(p+"\n")
        f.write("\n")
        if y:
            for i, p in enumerate(pred):
                total += 1
                if p == y[i]:
                    pred_correct += 1
    if total > 0:
        print("ADVANCED ACCURACY {}".format(pred_correct / total))
    f.close()
    return tagger


def parse_pos(pos):
    pos = re.sub("[\^]", " ", pos)
    pos = pos.strip()
    pos = re.sub("\\s+", " ", pos)
    return pos



if __name__ == '__main__':
    input_path = sys.argv[1]
    test_dir = sys.argv[2]
    output_file = sys.argv[3]
    data = get_data(input_path)
    X_features = []
    Y_features = []
    for dialog in data:
        x, y = create_features(dialog)
        X_features.append(x)
        if y:
            Y_features.append(y)
    trainer = create_trainer(X_features, Y_features)
    trainer.train('advanced.crfsuite')
    test = get_data(test_dir)
    tagger = evaulate_tagger(test, output_file)
