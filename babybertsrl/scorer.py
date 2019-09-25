"""
Code obtained from Allen AI NLP toolkit in September 2019
"""


import os
import shutil
import subprocess
import tempfile
from collections import defaultdict

from babybertsrl import config


class SrlEvalScorer:
    """
    This class uses the external srl-eval.pl script for computing the CoNLL SRL metrics.
    AllenNLP contains the srl-eval.pl script, but you will need perl 5.x.
    Note that this metric reads and writes from disk quite a bit. In particular, it
    writes and subsequently reads two files per __call__, which is typically invoked
    once per batch. You probably don't want to include it in your training loop;
    instead, you should calculate this on a validation set only.
    Parameters
    ----------
    ignore_classes : ``List[str]``, optional (default=``None``).
        A list of classes to ignore.
    """
    def __init__(self, ignore_classes=None):

        self._srl_eval_path = str(config.Eval.srl_eval_path)  # The path to the srl-eval.pl script.
        self._ignore_classes = set(ignore_classes)

        # These will hold per label span counts.
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def __call__(self,  # type: ignore
                 batch_verb_indices,  # : List[Optional[int]]
                 batch_sentences,  # : List[List[str]]
                 batch_conll_formatted_predicted_tags,  # : List[List[str]]
                 batch_conll_formatted_gold_tags) -> None:  # : List[List[str]]
        # pylint: disable=signature-differs
        """
        Parameters
        ----------
        batch_verb_indices : ``List[Optional[int]]``, required.
            The indices of the verbal predicate in the sentences which
            the gold labels are the arguments for, or None if the sentence
            contains no verbal predicate.
        batch_sentences : ``List[List[str]]``, required.
            The word tokens for each instance in the batch.
        batch_conll_formatted_predicted_tags : ``List[List[str]]``, required.
            A list of predicted CoNLL-formatted SRL tags (itself a list) to compute score for.
            Use allennlp.models.semantic_role_labeler.convert_bio_tags_to_conll_format
            to convert from BIO to CoNLL format before passing the tags into the metric,
            if applicable.
        batch_conll_formatted_gold_tags : ``List[List[str]]``, required.
            A list of gold CoNLL-formatted SRL tags (itself a list) to use as a reference.
            Use allennlp.models.semantic_role_labeler.convert_bio_tags_to_conll_format
            to convert from BIO to CoNLL format before passing the
            tags into the metric, if applicable.
        """
        if not os.path.exists(self._srl_eval_path):
            raise SystemError("srl-eval.pl not found at {}.".format(self._srl_eval_path))
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.txt")
        predicted_path = os.path.join(tempdir, "predicted.txt")

        with open(predicted_path, "w") as predicted_file, open(gold_path, "w") as gold_file:
            for verb_index, sentence, predicted_tag_sequence, gold_tag_sequence in zip(
                    batch_verb_indices,
                    batch_sentences,
                    batch_conll_formatted_predicted_tags,
                    batch_conll_formatted_gold_tags):
                write_conll_formatted_tags_to_file(predicted_file,
                                                   gold_file,
                                                   verb_index,
                                                   sentence,
                                                   predicted_tag_sequence,
                                                   gold_tag_sequence)
        perl_script_command = ["perl", self._srl_eval_path, gold_path, predicted_path]
        completed_process = subprocess.run(perl_script_command, stdout=subprocess.PIPE,
                                           universal_newlines=True, check=True)

        if config.Eval.verbose:
            print(completed_process.stdout)

        for line in completed_process.stdout.split("\n"):
            stripped = line.strip().split()
            if len(stripped) == 7:
                tag = stripped[0]
                # Overall metrics are calculated in get_metric, skip them here.
                if tag == "Overall" or tag in self._ignore_classes:
                    print('Skipping collection of tp, fp, and fn for tag={}'.format(tag))
                    continue
                # This line contains results for a span
                num_correct = int(stripped[1])
                num_excess = int(stripped[2])
                num_missed = int(stripped[3])
                self._true_positives[tag] += num_correct
                self._false_positives[tag] += num_excess
                self._false_negatives[tag] += num_missed
        shutil.rmtree(tempdir)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float
        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            if tag == "overall":
                raise ValueError("'overall' is disallowed as a tag type, "
                                 "rename the tag type to something else if necessary.")
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)


def convert_bio_tags_to_conll_format(labels):  # List[str]
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").
    A full example of the conversion performed:
    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]
    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.
    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels


def write_conll_formatted_tags_to_file(prediction_file,  # : TextIO
                                       gold_file,  # : TextIO
                                       verb_index,  # : Optional[int]
                                       sentence,  # : List[str]
                                       conll_formatted_predictions,  #: List[str]
                                       conll_formatted_gold_labels):  # : List[str]
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.
    The CoNLL SRL format is described in
    `the shared task data README <https://www.lsi.upc.edu/~srlconll/conll05st-release/README>`_ .
    This function expects IOB2-formatted tags, where the B- tag is used in the beginning
    of every chunk (i.e. all chunks start with the B- tag).
    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    conll_formatted_predictions : List[str], required.
        The predicted CoNLL-formatted labels.
    conll_formatted_gold_labels : List[str], required.
        The gold CoNLL-formatted labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_formatted_predictions,
                                     conll_formatted_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")