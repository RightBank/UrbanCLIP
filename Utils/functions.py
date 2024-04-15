def construct_uf_word_place_dict(uf_vocab_dict):
    count = 0
    uf_word_place_dict = {}
    function_name_list = list(uf_vocab_dict.keys())
    for uf, uot_list in uf_vocab_dict.items():
        uf_index = int(function_name_list.index(uf))
        for uot in uot_list:
            if uf_index not in uf_word_place_dict:
                uf_word_place_dict[uf_index] = []
            uf_word_place_dict[uf_index].append(count)
            count += 1
    return uf_word_place_dict

def label_conversion(num_label, categories):
    onehot_label = []
    for sample_label in categories:
        onehot = [0] * num_label
        for label in sample_label:
            onehot[int(label)-1] = 1
        onehot_label.append(onehot)
    return onehot_label