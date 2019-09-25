# -*- coding: utf-8 -*-
# @File    : image_caption/caption_generator.py
# @Info    : @ TSMC-SIGGRAPH, 2018/8/27
# @Desc    : refer to google/im2txt
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import heapq
import math

import numpy as np

from inference_wrapper import InferenceWrapper


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, attenplot=None):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          attenplot: Optional attention_weights plot associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.attenplot = attenplot

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


# 维护一个元素个数为n的堆,要么元素入堆,要么清空堆输出全部元素(安全起见,此时后面务必立刻接reset()方法)
class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)  # is equivalent to pushing first, then popping

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of dataset; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self, model: InferenceWrapper, beam_size=3, max_caption_length=20, length_normalization_factor=0.0):
        """Initializes the generator.
        Inputs:
            model: Object encapsulating a trained image-to-text model. Must have methods feed_image() and inference_step().
                    For example, an instance of InferenceWrapperBase.
            beam_size: Beam size to use when generating captions.
            max_caption_length: The maximum caption length before stopping the search.
            length_normalization_factor: If != 0, a number x such that captions are scored by logprob/length^x, rather than
            logprob. This changes the relative scores of captions depending on their lengths. For example, if x > 0 then
            longer captions will be favored.
        """
        self.model = model
        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, sess, encoded_image):
        """Runs beam search caption generation on a single image.
        Inputs:
            sess: TensorFlow Session object.
            encoded_image: An encoded image string.
        Returns:
            A list of Caption sorted by descending score.
        """
        # Feed in the image to get the image features.
        annotation_features_feed, init_state_feed = self.model.feed_image(sess, encoded_image)

        # initial_state = np.zeros(self.model.hidden_units)  # zero_state for GRU initial state
        # passing mean of annotation vectors to gru init_state
        # print("[beam_search] annotation_features_feed.shape: {}, "
        #       "\n\t\t init_state_feed.shape: {}".format(annotation_features_feed.shape, init_state_feed.shape))

        # note: sentence is a list of word ids in the caption.here vocab.start_id saves start_word ids when initializing.
        initial_beam = Caption(
            sentence=[self.model.tokenizer.word_index['<start>']],
            state=init_state_feed[0],
            logprob=0.0,
            score=0.0,
            attenplot=[])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)  # review: so far, only including init image and init string `<start>`
        complete_captions = TopN(self.beam_size)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()  # 获取起始序列,这个序列会随search循环不断延长
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])
            # print("[beam_search] input_feed.shape: {}, state_feed.shape: {}".format(input_feed.shape, state_feed.shape))
            # hint: inference_step will concatentate the resulting state.
            softmax, new_states, atten_data = self.model.inference_step(sess=sess, input_feed=input_feed,
                                                                        state_feed=state_feed,
                                                                        annotation_features_feed=annotation_features_feed)
            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]  # 提取当前时间步的预测
                state = new_states[i]  # 生成前一个词的预测后当前时间步的Cell_state
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])  # In probability, in descending order
                words_and_probs = words_and_probs[0:self.beam_size]
                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob

                    attendata_list = partial_caption.attenplot + [atten_data]

                    if w == self.model.tokenizer.word_index['<end>']:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, attendata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, attendata_list)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if not complete_captions.size():
            complete_captions = partial_captions

        return complete_captions.extract(sort=True)
