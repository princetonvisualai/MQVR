import os
import argparse
import copy
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

import src.model.model as module_arch
import src.data_loader as module_data
import src.model.metric as module_metric
from src.utils import *



def similarity_aggregation(query_text_features, all_video_features):
    sim_mat_list = [compute_sim_mat(query_text_features[:, idx, :], all_video_features) 
                            for idx in range(query_text_features.shape[1])]
    sim_mat = torch.stack(sim_mat_list, dim=0)
    sim_mat = sim_mat.mean(dim=0)
    return sim_mat


def rank_aggregation(query_text_features, all_video_features):
    sim_mat_list = [compute_sim_mat(query_text_features[:, idx, :], all_video_features)
                            for idx in range(query_text_features.shape[1])]
    seperate_retreival_res = [((-sim_mat).argsort(dim=1)).argsort() for sim_mat in sim_mat_list]
    ind = np.diag(sum(seperate_retreival_res).argsort(dim=1).argsort().cpu().numpy())
    return ind


def feature_aggregation(query_text_features, all_video_features, model):
    with torch.no_grad():
        text_features = model.text_head(query_text_features)
    sim_mat = compute_sim_mat(text_features, all_video_features)
    return sim_mat


def extract_features(model, eval_loader, tokenizer, device):
    all_video_features, all_text_features = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (frames, captions) in enumerate(tqdm(eval_loader)):
            frames = frames.to(device)
            captions = tokenizer.tokenize(captions)
            if isinstance(captions, torch.Tensor):
                captions = captions.to(device)
            else:
                captions = {k: v.to(device) for k, v in captions.items()}

            video_features, text_features = model.base(frames, captions)
            all_video_features.append(video_features)
            all_text_features.append(text_features)

    all_video_features = torch.cat(all_video_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    return all_video_features, all_text_features


def varying_query_eval(model, all_text_features, all_video_features, all_sample_idx, device):
    model.eval()
    result_dict = {
        'similarity_aggregation': {f'{n_query}_query': {} for n_query in all_sample_idx.keys()},
        'rank_aggregation': {f'{n_query}_query': {} for n_query in all_sample_idx.keys()},
        'feature_aggregation': {f'{n_query}_query': {} for n_query in all_sample_idx.keys()},
    }
    
    with torch.no_grad():
        for n_query, sample_idx in tqdm(all_sample_idx.items()):
            try:
                for idx in sample_idx:
                    idx = torch.tensor(idx).unsqueeze(2).expand(-1, -1, all_text_features.shape[2]).to(device)
                    query_text_features = torch.gather(all_text_features, 1, idx)

                    sim_mat = similarity_aggregation(query_text_features, all_video_features)
                    metrics = module_metric.retrieval_metric(sim_mat)
                    for key, item in metrics.items():
                        result_dict['similarity_aggregation'][f'{n_query}_query'].setdefault(key, []).append(item)

                    ind = rank_aggregation(query_text_features, all_video_features)
                    metrics = module_metric.compute_metric_from_rank(ind)
                    for key, item in metrics.items():
                        result_dict['rank_aggregation'][f'{n_query}_query'].setdefault(key, []).append(item)

                    sim_mat = feature_aggregation(query_text_features, all_video_features, model)
                    metrics = module_metric.retrieval_metric(sim_mat)
                    for key, item in metrics.items():
                        result_dict['feature_aggregation'][f'{n_query}_query'].setdefault(key, []).append(item)
            except:
                # on VATEX too many queries might result in out of memory error for some of the methods
                # could use cpu to get the result if needed but expect it to be slow
                result_dict['similarity_aggregation'][f'{n_query}_query'] = {}
                result_dict['rank_aggregation'][f'{n_query}_query'] = {}
                result_dict['feature_aggregation'][f'{n_query}_query'] = {}

    mean_result = {key: {f'{n_query}_query': {} for n_query in all_sample_idx.keys()} 
                        for key in result_dict.keys()}
    for method, result in result_dict.items():
        for n_query, n_query_res in result.items():
            for metric, metric_res in n_query_res.items():
                mean_result[method][n_query][metric] = np.mean(metric_res)

    return mean_result


def main(args):
    
    config = read_json(args.config)
    record_path = args.record_path
    if record_path is None:
        record_path = Path(config['trainer']['save_dir']) / 'models' \
                                / config['name'] / str(config.get('seed', 1))
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    test_loader_args = copy.deepcopy(config['data_loader']['args'])
    test_loader_args['video_setting']['sample'] = 'uniform'
    test_loader_args['text_setting']['sample'] = 'all'
    test_data_loader = getattr(module_data, config['data_loader']['type'])(
                               config['data_loader']['test_path'], 
                               shuffle=False, drop_last=False, 
                               **test_loader_args)

    tokenizer = Tokenizer(config['arch']['args']['base_setting']['type'])
    model = getattr(module_arch, config['arch']['type'])(
                    device=device, **config['arch']['args'])
    state_dict = torch.load(os.path.join(record_path, 'checkpoint-latest.pth'))
    
    # handle model saved in DataParallel mode
    if list(state_dict['state_dict'].keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k[7:]:v for k,v in state_dict['state_dict'].items()})
    else:
        model_state_dict = state_dict['state_dict']

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    print('Encoding video, text features...')
    test_video_features, test_text_features = extract_features(model, test_data_loader, tokenizer, device)

    print('Evaluating with different number of queries...')
    all_sample_idx = read_json(config['test_eval_sample'])
    val_result_dict = varying_query_eval(model, test_text_features, test_video_features, all_sample_idx, device)
    write_json(val_result_dict, os.path.join(record_path, 'test_varying_query_eval_result.json'))



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--record_path', default=None, type=str)
    args.add_argument('--cpu', dest='cuda', action='store_false')
    args.set_defaults(cuda=True)
    args = args.parse_args()

    main(args)


