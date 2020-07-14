import os,sys
from hw2_corpus_tool import get_data
import pycrfsuite as crf
SPEAKER = 'speaker'
ACT_TAG = 'act_tag'
BOD = 'BEGINNING_OF_DIALOG'


def check_speaker_change(prev_speaker,current_speaker):
    if prev_speaker and not prev_speaker == current_speaker:
        return True
    return False


def create_pos_tag_list(pos_list):
    result = []
    if not pos_list or (len(pos_list) == 1 and pos_list[0][0] == '.' and pos_list[0][1] == '.'):
        return ['NO_WORDS']
    for tup in pos_list:
        result.append('TOKEN_'+tup[0])
        result.append('POS_' + tup[1])

    # result.extend(token_result)
    return result


def create_features(dialog):
    dialog_features = []
    labels = []
    prev_speak = None
    for i, utterance in enumerate(dialog):
        features = []
        if ACT_TAG in utterance._fields:
            labels.append(getattr(utterance,ACT_TAG))
        postag_list =getattr(utterance,'pos')
        result= create_pos_tag_list(postag_list)
        features.extend(result)
        if i>0:
            if not prev_speak == getattr(utterance,SPEAKER):
                features.append('SPEAKER_CHANGE')
        else:
           features.append(BOD)
        prev_speak = getattr(utterance,SPEAKER)
        dialog_features.append(features)
    return dialog_features, labels

def create_trainer(X,Y):
    trainer = crf.Trainer()
    for x,y in zip(X,Y):
        trainer.append(x,y)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    return trainer

def  evaulate_base_tagger(data,outputfile):
    tagger = crf.Tagger()
    tagger.open('base.crfsuite')
    total = 0
    pred_correct = 0
    f = open(outputfile,'w')
    for dialog in data:
        x, y = create_features(dialog)
        pred = (tagger.tag(x))
        for p in pred:
            f.write(p+"\n")
        if y:
            for i, p in enumerate(pred):
                total += 1
                if p == y[i]:
                    pred_correct += 1
    if total > 0:
        print(pred_correct / total)
    f.close()


if __name__ == '__main__':
    # input_path = 'train/'
    test_dir = 'dev/'
    output_file = 'op.txt'
    # data = get_data(input_path)
    # X_features = []
    # Y_features = []
    # count = 0
    # for dialog in data:
    #     x, y = create_features(dialog)
    #     count += len(x)
    #     X_features.append(x)
    #     if y:
    #         Y_features.append(y)
    # trainer = create_trainer(X_features, Y_features)
    # trainer.train('base.crfsuite')
    test = get_data(test_dir)
    evaulate_base_tagger(test,output_file)


