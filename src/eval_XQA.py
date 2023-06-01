import argparse, os
import jsonlines
from parse import parse
import json

def parse_prediction(prediction):
    # prediction = prediction.upper()
    #! 不管是 4 个选项还是 5 个选项，都不影响
    if 'A' in prediction:
        for choice in ['B', 'C', 'D', 'E']:
            if choice in prediction:
                return None
        return 'A'

        
    
    elif 'B' in prediction:
        for choice in ['A', 'C', 'D', 'E']:
            if choice in prediction:
                return None
        return 'B'


    elif 'C' in prediction:
        for choice in ['A', 'B', 'D', 'E']:
            if choice in prediction:
                return None
        return 'C'

    elif 'D' in prediction:
        for choice in ['A', 'B', 'C', 'E']:
            if choice in prediction:
                return None
        return 'D'
        
    elif 'E' in prediction:
        for choice in ['A', 'B', 'C', 'D']:
            if choice in prediction:
                return None
        return 'E'
    else:
        return None
def parse_prediction_v2(prediction):
    if 'Question' in prediction:
        prediction = prediction.split('Question')[0].strip()
    elif 'Qestion' in prediction:
        prediction = prediction.split('Qestion')[0].strip()
    elif 'Premise' in prediction:
        prediction = prediction.split('Premise')[0].strip()
        
    return parse_prediction(prediction)
    

def get_metric(file_path, response_key):
    with jsonlines.open(file_path, 'r') as f:
        valid_count = 0; correct_count = 0
        for idx, line in enumerate(f):
            answer_key = line['answerKey']
            prediction = line[response_key]
            #! print
            # if 'use_example' in file_path and 'zh' in file_path:
            #     print(f'Instruction:{line["instruction"]}')
            #     print(f'AnswerKey:{line["answerKey"]}')
            #     print(f'Response:{prediction}')
            #     print()
            #     print()
            #     import pdb; pdb.set_trace()
            #! use_exp need to use v2
            prediction = parse_prediction_v2(prediction)
            
           
            
            if prediction:
                valid_count += 1
                if prediction == answer_key:
                    correct_count += 1
        # if 'use_example' in file_path and '/zh/' in file_path:
        #     print(file_path)
        #     print(correct_count)
        try:
            acc = round(correct_count / valid_count, 4)
        except ZeroDivisionError:
            return dict(accuracy=None, valid_ratio=0)
        valid_ratio = round(valid_count / (idx+1), 4)
        return dict(accuracy=acc, valid_ratio=valid_ratio)
                    
                
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default=None, type=str, required=True)
    parser.add_argument('--metric_dir', default=None, type=str, required=True)
    args = parser.parse_args()
    
    metric_dict = dict()
    
    for lang in os.listdir(args.result_dir):
        lang_dir = os.path.join(args.result_dir, lang)
        if lang not in metric_dict.keys():
            metric_dict[lang] = {}
        
        for add_lang_prompt in os.listdir(lang_dir):
            
            add_lang_prompt_dir = os.path.join(lang_dir, add_lang_prompt)
            if add_lang_prompt not in metric_dict[lang].keys():
                metric_dict[lang][add_lang_prompt] = {}
                
            for prompt_type in os.listdir(add_lang_prompt_dir):
                
                prompt_type_dir = os.path.join(add_lang_prompt_dir, prompt_type)
                if prompt_type not in metric_dict[lang][add_lang_prompt].keys():
                    metric_dict[lang][add_lang_prompt][prompt_type] = {}
                
                
                for example_type in os.listdir(prompt_type_dir):
                    example_type_dir = os.path.join(prompt_type_dir, example_type)
                    
                    if example_type not in metric_dict[lang][add_lang_prompt][prompt_type].keys():
                        metric_dict[lang][add_lang_prompt][prompt_type][example_type] = {}
                    
                    for file in os.listdir(example_type_dir):
                        
                    
                        parse_result = parse('{model_type}-for-{data_type}', file)
                        # import pdb; pdb.set_trace()
                        model_type = parse_result['model_type']
                        
                        if model_type not in metric_dict[lang][add_lang_prompt][prompt_type][example_type].keys():
                            metric_dict[lang][add_lang_prompt][prompt_type][example_type][model_type] = {}
                            
                            
                        response_key = model_type + '_response'
                        file_path = os.path.join(example_type_dir, file)
                        
                        cur_metric_dict = get_metric(file_path, response_key)
                        metric_dict[lang][add_lang_prompt][prompt_type][example_type][model_type].update(cur_metric_dict)
                
    metric_file_path = os.path.join(args.metric_dir, 'metric.json')
    json.dump(metric_dict, open(metric_file_path, 'w+'), indent=4)
    
    
    
    
    
