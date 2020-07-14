# NLP-CRFSuite-Example
USC - NLP Assignment on sequence labelling

### Data used for this task - Switchboard corpus

### Data input (Dialogue Utterance) to the models (baseline and advanced) must be in the format:
A named tuple having the following fields:
-  act_tag - the dialog act associated with this utterance
-  speaker - which speaker made this utterance
-  pos - a list of PosTag objects (token and POS)
-  text - the text of the utterance with only a little bit of cleaning"""

### Implementation of a BaseLine Tagger ###

The BaseLine Tagger contains the following features for each dialogue utterance in in the sequence:
 - pos tags of the utterance
 - tokens in the text
 - whether the utterance is the beginning of a dialogue
 - whether the speaker has changed in the dialogue
 
 
### Implementation of a Advanced Tagger ###

The advanced tagger contains the features in baseline along with the following:
- POS tags and tokens of current utterance
- POS tags and tokens of previous utterance
- Bigrams of pos tags and tokens of current utterance

#### Accuracy of baseline tagger : 0.722
#### Accuracy of Advanced tagger: 0.733
