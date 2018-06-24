# -*- coding: utf-8 -*-
# @File    : tryEverything/beam_search.py
# @Info    : @ TSMC-SIGGRAPH, 2018/6/19
# @Desc    : refer to google/im2txt
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import heapq
import math

import numpy as np


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score

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

    def __init__(self,
                 model,
                 vocab,
                 new_state,
                 beam_size=3,
                 max_caption_length=50):
        """Initializes the generator.

        Args:
          model: Object encapsulating a trained image-to-text model. Must have
            methods feed_image() and inference_step(). For example, an instance of
            InferenceWrapperBase.
          vocab: A Vocabulary object.
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
        """
        self.vocab = vocab
        self.model = model

        self.new_state = new_state

        self.beam_size = beam_size
        self.max_caption_length = max_caption_length

    def get_initial_beam_state(self, sess, start_words):
        # new_state = sess.run(self.model.initial_state)
        new_state = self.new_state
        start_ids = [self.vocab.word_to_id(word) for word in start_words]
        x = np.zeros((1, 1))  # because we have set sampling=True, means batch_size,n_seqs=1,1
        logprob = 0.0
        for ids in start_ids:
            x[0, 0] = ids  # input one word
            word_probs, new_state = self.inference(sess, x, new_state)
            if word_probs[ids] < 1e-12:
                continue  # Avoid log(0).
            logprob = logprob + math.log(word_probs[ids])
        return logprob, new_state, start_ids[-1]

    def inference(self, sess, input_feed, state_feed):
        feed = {self.model.inputs: input_feed, self.model.keep_prob: 1., self.model.initial_state: state_feed}
        word_probs, new_state = sess.run([self.model.prediction, self.model.final_state], feed_dict=feed)
        return word_probs[0], new_state

    def beam_search(self, sess, start_words):
        init_logprob, init_state, init_ids = self.get_initial_beam_state(sess, start_words)

        partial_captions = TopN(self.beam_size)
        initial_beam = Caption(sentence=[init_ids], state=init_state, logprob=init_logprob, score=init_logprob)
        partial_captions.push(initial_beam)

        complete_captions = TopN(self.beam_size)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()  # 获取起始序列,这个序列会随search循环不断延长
            partial_captions.reset()
            wp_list = list()
            new_states = list()
            for cur_seq_tag, partial_caption in enumerate(partial_captions_list):
                input_feed = np.array([partial_caption.sentence[-1]]).reshape((1, 1))
                state_feed = partial_caption.state[-1]

                # hint: inference_step
                word_probs, new_state = self.inference(sess, input_feed, state_feed)

                new_states.append(new_state)
                words_and_probs = list(enumerate(word_probs))
                words_and_probs.sort(key=lambda x: -x[1])  # In probability, in descending order
                words_and_probs = words_and_probs[0:self.beam_size]
                for next_idx, prob in words_and_probs:
                    if prob < 1e-12:
                        wp_list.append(
                            (cur_seq_tag, next_idx, partial_caption.logprob))  # [cur_seq_tag, next_idx,logprob]
                    else:
                        wp_list.append(
                            (cur_seq_tag, next_idx, partial_caption.logprob + math.log(prob)))

            wp_list.sort(key=lambda x: -x[2])
            wp_list = wp_list[0:self.beam_size]

            for cur_seq_tag, w, p in wp_list:
                sentence = partial_captions_list[cur_seq_tag].sentence + [w]
                logprob = p
                score = logprob
                beam = Caption(sentence, new_states[cur_seq_tag], logprob, score)
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
