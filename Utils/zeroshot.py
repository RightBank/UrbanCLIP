import numpy as np
import torch
import clip
from tqdm import tqdm
from itertools import chain
import pickle
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
import json
from Utils.prompt_templates import *
from Utils.functions import *


def zeroshot_classifier(model, device, logit_scale, classnames, templates, image_features, laion_features, ensemble):
    with torch.no_grad():
        zeroshot_weights = []
        if ensemble == 'mean':
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device).to(torch.float32)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            logits = logits.softmax(dim=-1)

        elif ensemble == 'zpe':
            class_emb = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts).float()
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_emb.append(class_embeddings)
            class_emb = torch.stack(class_emb).transpose(1, 0).to(device).to(torch.float32)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_list = []
            sp_list = []

            laion_features = torch.tensor(list(laion_features.values())).to(device).to(torch.float32)
            laion_features /= laion_features.norm(dim=-1, keepdim=True)
            for template_logits in class_emb:
                test_text_logits = image_features @ template_logits.t() * logit_scale
                e_test = test_text_logits.mean(0)

                pretrain_text_logits = laion_features @ template_logits.t() * logit_scale
                e_pretrain = pretrain_text_logits.mean(0)
                logits_normalized = test_text_logits - 0.5 * (e_test + e_pretrain)
                max_logits = logits_normalized.max(dim=1)[0]
                s_p = max_logits.mean()
                sp_list.append(s_p)
                logits_list.append(test_text_logits)
            softmax_sp = torch.softmax(torch.tensor(sp_list), dim=0).to(device).to(torch.float32)
            logits = (softmax_sp.unsqueeze(1).unsqueeze(1) * torch.stack(logits_list)).mean(0)
    return logits


def zeroshot_inference (args):
    device = args.device
    prompt_template = args.prompt_template
    task = args.task
    ensemble = args.ensemble
    taxonomy = args.taxonomy
    print("Testing on the scenario: ", task, "\n using the template: ", prompt_template, "\n using",
          taxonomy, "as class names",  "\n ensemble: ", ensemble)

    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    logit_scale = model.logit_scale


    if prompt_template == 'UrbanCLIP':
        svi_templates = urbanclip_templates
    elif prompt_template == 'CLIP80':
        svi_templates = openai80_templates
    elif prompt_template == 'Photo':
        svi_templates = photo_templates
    elif prompt_template == 'no_template':
        svi_templates = [
            '{}',
        ]
    elif prompt_template == 'UrbanCLIP_SC':
        if task == 'primary' or task == 'multi':
            svi_templates = urbanclip_templates_sc_shenzhen
        elif task == 'transfer-london':
            svi_templates = urbanclip_templates_sc_london
        elif task == 'transfer-singapore':
            svi_templates = urbanclip_templates_sc_singapore
    elif prompt_template == 'Wu':
        if task == 'primary' or task == 'multi':
            svi_templates = wu_templates_shenzhen
        elif task == 'transfer-london':
            svi_templates = wu_templates_london
        elif task == 'transfer-singapore':
            svi_templates = wu_templates_singapore
    elif prompt_template == 'Wu_without_SC':
        svi_templates = wu_templates_no_sc

    uf_vocab_dict = json.load(open('./Utils/urban_taxonomy.json', 'r'), object_pairs_hook=OrderedDict)
    word_list = list(chain.from_iterable(list(uf_vocab_dict.values())))

    if taxonomy == 'function_name':
        uf_vocab_dict = dict(zip(list(uf_vocab_dict.keys()), list(uf_vocab_dict.keys())))
        word_list = list(uf_vocab_dict.keys())

    if task == 'primary' or task == 'multi':
        image_features = torch.load('./Emb/clip').to(torch.float32).to(device)
    elif task == 'transfer-london':
        image_features = torch.load('./Emb/clip_london').to(torch.float32).to(device)
    elif task == 'transfer-singapore':
        image_features = torch.load('./Emb/clip_singapore').to(torch.float32).to(device)

    with open('./Utils/laion_emb_dict1.pkl', 'rb') as handle:
        laion_emb_dict1 = pickle.load(handle)
    with open('./Utils/laion_emb_dict2.pkl', 'rb') as handle:
        laion_emb_dict2 = pickle.load(handle)
    laion_emb_dict = {**laion_emb_dict1, **laion_emb_dict2}

    logits = zeroshot_classifier(model=model, device=device, logit_scale=logit_scale, classnames=word_list,
                                    templates=svi_templates,
                                 image_features=image_features, laion_features=laion_emb_dict, ensemble=ensemble)

    uf_word_place_dict = construct_uf_word_place_dict(uf_vocab_dict)

    top_1_list = []
    if task == 'multi':
        top_2_list = []

    with torch.no_grad():
        for i in tqdm(range(image_features.shape[0])):
            max_logit_per_func = {}
            if taxonomy == 'UrbanCLIP':
                for uf_index, uf_places in uf_word_place_dict.items():
                    max_logit_per_func[uf_index] = logits[i][uf_places].max().item()
                top_1_list.append(max(max_logit_per_func, key=max_logit_per_func.get) + 1)
                if task == 'multi':
                    top_2_list.append(sorted(max_logit_per_func, key=max_logit_per_func.get)[-2] + 1)
            elif taxonomy == 'function_name':
                top_1_list.append(logits[i].argmax().item() + 1)
                if task == 'multi':
                    top_2_list.append(np.argsort(logits[i].detach().cpu().numpy())[-2] + 1)
    if task == 'primary':
        image_list = pd.read_csv('./Data/Urban_scene_dataset_Shenzhen/image_list.csv', encoding="ISO-8859-1")
        inferred_funcs = np.asarray(top_1_list)
        ground_truth = np.asarray(image_list['primary_function'].tolist())

    elif task == 'multi':
        image_list = pd.read_csv('./Data/Urban_scene_dataset_Shenzhen/image_list.csv', encoding="ISO-8859-1")
        multi_image_list = image_list[image_list['secondary_function'].notnull()]
        multi_ground_truth = list(zip(multi_image_list['primary_function'].tolist(), multi_image_list['secondary_function'].tolist()))
        ground_truth = label_conversion(10, multi_ground_truth)

        inferred_funcs = list(zip(top_1_list, top_2_list))
        inferred_funcs = np.asarray(label_conversion(10, inferred_funcs))
        inferred_funcs = inferred_funcs[image_list['secondary_function'].notnull()]

    elif task == 'transfer-london':
        image_list = pd.read_csv('./Data/Urban_scene_dataset_London/london_transfer_images.csv',
                                     encoding="ISO-8859-1")
        inferred_funcs = np.asarray(top_1_list)
        ground_truth = np.asarray(image_list['primary_function'].tolist())

    elif task == 'transfer-singapore':
        image_list = pd.read_csv('./Data/Urban_scene_dataset_Singapore/singapore_transfer_images.csv',
                                     encoding="ISO-8859-1")
        inferred_funcs = np.asarray(top_1_list)
        ground_truth = np.asarray(image_list['primary_function'].tolist())

    acc_average = accuracy_score(ground_truth, inferred_funcs)
    f1_average = f1_score(ground_truth, inferred_funcs, average='weighted')

    print("Results")
    print('acc: ', acc_average, 'f1: ', f1_average)
    if task == 'multi':
        jaccard = jaccard_score(ground_truth, inferred_funcs, average='weighted', zero_division=0)
        print('jaccard: ', jaccard)
















