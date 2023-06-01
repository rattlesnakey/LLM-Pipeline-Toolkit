from rouge import Rouge
import argparse, os
import jsonlines
from parse import parse
import json

def is_Chinese(sen):
    for word in sen:
        if '\u4e00' <= word <= '\u9fff':
            return True
    return False



def get_rouge(text, ref_text):
    if is_Chinese(ref_text):
        text = ' '.join(list(text))
        ref_text = ' '.join(list(ref_text))
        # return None, None, None
        
    text = [text]
    ref_text = [ref_text]
    try:
        rouge_score = rouge.get_scores(text, ref_text)
        rouge_1 = rouge_score[0]["rouge-1"]['f']
        rouge_2 = rouge_score[0]["rouge-2"]['f']
        rouge_l = rouge_score[0]["rouge-l"]['f']
        # import pdb; pdb.set_trace()
    except ValueError:
        return None, None, None
    return rouge_1, rouge_2, rouge_l

def get_metric(file_path, response_key):
    # import pdb; pdb.set_trace()
    with jsonlines.open(file_path, 'r') as f:
        valid_count = 0
        total_rouge_1 = 0; total_rouge_2 = 0; total_rouge_l = 0
        for idx, line in enumerate(f):
            rouge_1, rouge_2, rouge_l = get_rouge(line[response_key], line['output'])
            
            if rouge_1 != None:
                valid_count += 1
                total_rouge_1 += rouge_1
                total_rouge_2 += rouge_2
                total_rouge_l += rouge_l
        result = dict(
            rouge_1=round((total_rouge_1 / valid_count), 4),
            rouge_2=round((total_rouge_2 / valid_count), 4),
            rouge_l=round((total_rouge_l / valid_count), 4),  
            valid_ratio=valid_count / (idx+1)
        )
        return result
                
                
            
        
                
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default=None, type=str, required=True)
    parser.add_argument('--metric_dir', default=None, type=str, required=True)
    args = parser.parse_args()
    rouge = Rouge()
    metric_dict = dict()
    for add_lang in os.listdir(args.result_dir):
        add_lang_dir = os.path.join(args.result_dir, add_lang)
        for promp_type in os.listdir(add_lang_dir):
            prompt_type_dir = os.path.join(add_lang_dir, promp_type)
            for exp_type in os.listdir(prompt_type_dir):
                exp_type_dir = os.path.join(prompt_type_dir, exp_type)
                
                for file in os.listdir(exp_type_dir):
                    file_path = os.path.join(exp_type_dir, file)
                    # import pdb; pdb.set_trace()
                    parse_result = parse('{model_type}-for-{data_type}', file)
                    # import pdb; pdb.set_trace()
                    model_type = parse_result['model_type']
                                    
                    if model_type not in metric_dict.keys():
                        metric_dict[model_type] = {}
                        
                        
                    response_key = model_type + '_response'
                    cur_metric_dict = get_metric(file_path, response_key)
                    metric_dict[model_type].update(cur_metric_dict)
                
    metric_file_path = os.path.join(args.metric_dir, 'metric.json')
    json.dump(metric_dict, open(metric_file_path, 'w+'), indent=4)
    