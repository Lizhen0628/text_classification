import argparse
import collections
import torch
import numpy as np
from tqdm import tqdm
# import data_process.data_loaders as module_data
import sys 
sys.path.append("..") 
from data_process import weibo_data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
from base import base_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.models as module_arch
from torch.functional import F
import torch.serialization
# 导入 ConfigParser (这是你缺失的)
# 请根据你的项目结构确认此路径正确
from utils.parse_config import ConfigParser 
# ===================== 修改点 1: 添加 sklearn.metrics =====================
from sklearn.metrics import accuracy_score, classification_report
# =========================================================================


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('test')
    # logger = config.get_logger('train') #<-- 这一行重复了，可以注释掉
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['num_gpu'] > 0 else 'cpu')

    # setup data_set, data_process instances
    test_dataset = config.init_obj('test_set', module_data_process)
    # setup data_loader instances
    
    try:
        batch_size = config['test_set']['args']['batch_size']
        num_workers = config['test_set']['args']['num_workers']
    except KeyError:
        batch_size = config['global_parameters']['batch_size']
        num_workers = config['global_parameters']['num_workers']

    
    logger.info(f"Manually creating test_dataloader with batch_size={batch_size}")
    test_dataloader = module_dataloader.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=test_dataset.collate_fn  
    )

    # build model architecture, then print to console
    logger.info("Initializing model architecture (model_arch)...")
    model = config.init_obj('model_arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    torch.serialization.add_safe_globals([ConfigParser])
    checkpoint = torch.load(config.resume,weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['num_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device) 
    model.eval()

    id_map_label = {0:'消极 (Negative)', 1:'积极 (Positive)'}
    
    # ===================== 修改点 2: 初始化空列表 =====================
    # 用于收集所有的预测值和真实标签
    all_preds = []
    all_labels = []
    # ==================================================================

    # inference
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader, desc="Inference")):
            # 确保你的 collate_fn_4_inference 确实返回5个值
            input_token_ids, attention_masks, seq_lens, class_labels, texts = batch_data
            
            # 将需要的数据移动到 device
            input_token_ids = input_token_ids.to(device)
            if attention_masks is not None:
                attention_masks = attention_masks.to(device)
            # seq_lens 在 RNN 模型中通常在 CPU 上处理 (如果它被用于 pack_padded_sequence)
            # 如果你的模型实现不需要，可以注释掉

            output, _ = model(input_token_ids, attention_masks, seq_lens)
            
            output_probs = F.softmax(output, dim=-1)
            output_preds = torch.argmax(output_probs, dim=-1)
            
            output_preds = output_preds.cpu().detach().numpy()
            class_labels = class_labels.cpu().detach().numpy()
            
            # ===================== 修改点 3: 填充列表 =====================
            # 将当前批次的结果添加到总列表中
            all_preds.extend(output_preds)
            all_labels.extend(class_labels)
            # ==============================================================

            # 打印每个样本的详细信息 (这部分可以保留，也可以注释掉)
            for text, pred, label in zip(texts, output_preds, class_labels):
                print('text:{}\npredict label:{}\ntrue label:{}'.format(text, id_map_label[pred], id_map_label[label]))
                print('--'*50)

    # ===================== 修改点 4: 计算并打印最终指标 =====================
    # 循环结束后，计算总体准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 获取标签名称用于分类报告
    label_names = [id_map_label[i] for i in sorted(id_map_label.keys())]
    
    # 生成详细的分类报告 (包括 P, R, F1-score)
    report = classification_report(all_labels, all_preds, target_names=label_names)

    print("\n" + "="*50)
    print(" 最终测试结果 ".center(50, "="))
    print("="*50)
    print(f"\n总样本数: {len(all_labels)}")
    print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
    
    print("\n详细分类报告 (Classification Report):")
    print(report)
    print("="*50)
    # =======================================================================


def run(config_file, model_path):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=model_path, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    parsed_args = args.parse_args()
    
    config_dict = vars(parsed_args)
    
    config = ConfigParser.from_args(config_dict)

    print(config.config['model_arch']['type'].lower())

    main(config)


if __name__ == '__main__':
    
    config_path = 'configs/binary_classification/word_embedding_rnn.yml' 
    model_checkpoint_path = 'saved/models/RnnModel/1104_173043/model_best.pth' 

    run(config_path, model_checkpoint_path)