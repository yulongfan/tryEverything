##  word-level-text-generation (chinese language)  ##
refer to https://github.com/NELSONZHAO/zhihu.git

zhihu/anna_lstm

##  Note    ##
requirements:
  * tqdm
  * tensorflow >=1.4
  * python3 >=3.5

only support python3

##  Train  ##
    python generate_chinese_text.py --is_training=True

##  Test  ##
    python generate_chinese_text.py --is_training=False