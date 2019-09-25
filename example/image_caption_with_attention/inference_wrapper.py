# -*- coding: utf-8 -*-
# @File    : derain_reinfore/inference_wrapper.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/12
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


from image_cap_model import ImageCapModel
from utils import inference_wrapper_base


class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
    """Model wrapper class for performing inference with a ShowAndTellModel."""

    def __init__(self):
        super(InferenceWrapper, self).__init__()
        # mapping ( word -> index )
        self.tokenizer = None
        # mapping ( index -> word )
        self.index_word = None
        self.hidden_units = None

    def build_model(self, model_config):
        model = ImageCapModel(model_config, mode="inference")
        model.build()
        self.tokenizer = model.cap_model_database.tokenizer
        self.index_word = model.cap_model_database.index_word
        self.hidden_units = model.config.units
        return model

    def feed_image(self, sess, encoded_image):
        # via image_feed pass image, after run, will get annotation_features with shape = (1, 64, 256)
        annotation_features, zero_state = sess.run(fetches=["imgcap/annotation_features:0",
                                                            "imgcap/mean_annotation_features:0"],
                                                   feed_dict={"image_feed:0": encoded_image})
        return annotation_features, zero_state

    def inference_step(self, sess, input_feed, state_feed, annotation_features_feed):
        # todo: ValueError: Cannot feed value of shape (1, 1, 512) for Tensor 'imgcap/state_feed:0', which has shape '(?, 512)'
        softmax_output, state_output, attention_weights = sess.run(
            fetches=["softmax:0", "imgcap/state:0", "imgcap/attention_weights:0"],
            feed_dict={
                "input_feed:0": input_feed,
                "imgcap/state_feed:0": state_feed,
                "imgcap/annotation_features_feed:0": annotation_features_feed,
            })
        return softmax_output, state_output, attention_weights
