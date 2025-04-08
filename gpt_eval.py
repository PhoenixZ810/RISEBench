import requests
import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable, Sized
import time
import re
from utils import *


api_key = 'YOUR_API_KEY'
api_base = 'YOUR_API_BASE'


def gpt_generate(inputs, model='gpt-4o-2024-11-20', temperature=0, max_tokens=4096, image_size=768, **kwargs):
    input_msgs = prepare_inputs(inputs)
    temperature = kwargs.pop('temperature', temperature)
    max_tokens = kwargs.pop('max_tokens', max_tokens)
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    payload = dict(
        model=model,
        messages=input_msgs,
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
        **kwargs)
    response = requests.post(
        api_base,
        headers=headers, data=json.dumps(payload), timeout=60)
    ret_code = response.status_code
    ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
    answer = 'Failed to obtain answer via API. '
    try:
        resp_struct = json.loads(response.text)
        answer = resp_struct['choices'][0]['message']['content'].strip()
    except Exception as err:
        print(f'{type(err)}: {err}')
        print(response.text if hasattr(response, 'text') else response)

    return ret_code, answer, response

def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
                    if keys is not None:
                        res[keys[idx]] = results[idx]
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    return results

def find_image(output_dir, index):
    for suffix in ['png', 'jpg', 'jpeg']:
        img_path = osp.join(output_dir, f"{index}.{suffix}")
        if osp.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Cannot find output images {index} in {output_dir}!!!")

def eval_vanilla(item, input_dir, output_dir, **kwargs):
    instruct = item['instruction']
    index = item['index']
    category = item['category']
    output_dir = osp.join(output_dir, f'images/{category}')
    img2 = find_image(output_dir, index)

    if category in ['temporal_reasoning', 'causal_reasoning']:
        img1 = osp.join(input_dir, item['image'])
        reference = item['reference']
        prompt1 = prompt_consist.format(instruct=instruct, reference=reference)
        prompt2 = prompt_reasoning.format(instruct=instruct, reference=reference)
        prompt3 = prompt_generation

    elif category == 'spatial_reasoning':
        img1 = osp.join(input_dir, item['image'])
        reference = item['reference']
        prompt1 = prompt_spatial_cons.format(instruct=instruct, reference=reference)
        prompt2 = prompt_spatial_ref.format(instruct=instruct, reference=reference)
        prompt3 = prompt_spatial_qual

    elif category == 'logical_reasoning':
        if "reference_txt" in item and not pd.isna(item['reference_txt']):
            img1 = osp.join(input_dir, item['image'])
            reference = item['reference_txt']
            prompt1 = prompt_logical_txt.format(instruct=instruct, reference=reference)
        elif "reference_img" in item and not pd.isna(item['reference_img']):
            img1 = osp.join(input_dir, item['reference_img'])
            prompt1 = prompt_logical_img.format(instruct=instruct)

    message = []
    text = {'type': 'text', 'value': prompt1}
    image1 = {
        'type': 'image',
        'value': img1,
    }
    image2 = {
        'type': 'image',
        'value': img2,
    }

    message.append(text)
    message.append(image1)
    message.append(image2)
    print(message)

    ret_code, consist_judge, response = gpt_generate(message, **kwargs)
    print(consist_judge)

    if category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        message2 = [{'type': 'text', 'value': prompt2}, {
            'type': 'image',
            'value': img2,
        }]
        ret_code2, answer2, response2 = gpt_generate(message2)

        message3 = [{'type': 'text', 'value': prompt3}, {
            'type': 'image',
            'value': img2,
        }]
        ret_code3, answer3, response3 = gpt_generate(message3)

        return dict(judge1=consist_judge, judge2=answer2, judge3=answer3)
    else:
        return dict(judge1=consist_judge)


def extract(answer):
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return numbers

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return numbers
    else:
        return None


def calculate_score(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        score = 0.4 * row['ImageConsistency'] + 0.4 * row['Reasoning'] + 0.2 * row['GenerationQuality']

    elif row['category'] == 'logical_reasoning':
        score = 0.4 * row['ImageConsistency'] + 0.6 * row['Reasoning']
    return score

def calculate_completion(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        return (
            1
            if row['ImageConsistency'] == 5 and row['Reasoning'] == 5 and row['GenerationQuality'] == 5
            else 0
        )
    elif row['category']=='logical_reasoning':
        return (
            1 if row['ImageConsistency'] == 5 and row['Reasoning'] == 5 else 0
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Json Path')
    parser.add_argument('--output', type=str, required=True, help='Output Image Dir, outputs/MODEL_NAME')
    parser.add_argument('--input', type=str, default='data', help='Input Image Dir')
    parser.add_argument('--prefix', type=str, default=None, help='output json prefix')
    parser.add_argument('--model', type=str, default=None, help='Model Name')
    parser.add_argument('--nproc', type=int, default=4, help='n processes for api')

    args = parser.parse_args()

    model_name = args.output.split('/')[-1] if args.model is None else args.model
    if not args.prefix:
        tmp_file = f"{args.output}/{model_name}.pkl"
        judge_res = f"{args.output}/{model_name}_judge.xlsx"
        score_file = f"{args.output}/{model_name}_judge.csv"
    else:
        tmp_file = f"{args.output}/{args.prefix}_{model_name}.pkl"
        judge_res = f"{args.output}/{args.prefix}_{model_name}_judge.xlsx"
        score_file = f"{args.output}/{args.prefix}_{model_name}_judge.csv"

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    data = json.load(open(args.data))
    data = pd.DataFrame(data)

    result = {}
    if osp.exists(tmp_file):
        result = load(tmp_file)

    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(item=x, input_dir=args.input, output_dir=args.output) for x in items]
    keys = [x['index'] for x in items]
    # breakpoint()
    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=args.nproc, chunksize=args.nproc, save=tmp_file, keys=keys)
        result = load(tmp_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v

    judges = [result[i] for i in data['index']]

    scores, judge_combine = [], []

    for judge in judges:
        if 'judge2' not in judge:
            judge_combine.append(judge['judge1'])
            score = [extract(judge['judge1'])[1], extract(judge['judge1'])[0]]
        elif 'judge3' not in judge:
            judge_combine.append('CONSISTENCY\n\n'+judge['judge1']+'\n\nREASOINING & QUALITY\n\n'+judge['judge2'])
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2
        else:
            judge_combine.append(
                'CONSISTENCY\n\n'
                + judge['judge1']
                + '\n\nREASOINING\n\n'
                + judge['judge2']
                + '\n\nQUALITY\n\n'
                + judge['judge3']
            )
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2+score3
        scores.append(score)

    reasoning = []
    img_consist = []
    gen_quality = []
    match_log = []

    for score in scores:
        if score:
            match_log.append('succeed')
            if len(score)==3:
                img_consist.append(score[0])
                reasoning.append(score[1])
                gen_quality.append(score[2])

            elif len(score)==2:
                reasoning.append(4 * min(score[1], 1) + 1)
                img_consist.append(4 * min(score[0], 1) + 1)
                gen_quality.append(None)
        else:
            img_consist.append(None)
            reasoning.append(None)
            gen_quality.append(None)
            match_log.append('failed')
    # breakpoint()
    data['Reasoning'] = reasoning
    data['ImageConsistency'] = img_consist
    data['GenerationQuality'] = gen_quality
    data['match_log'] = match_log
    data['judge'] = judge_combine

    data['score'] = data.apply(calculate_score, axis=1)
    data['complete'] = data.apply(calculate_completion, axis=1)

    dump(data, judge_res)

    df_causal = data[data['category'] == 'causal_reasoning']
    df_temporal = data[data['category'] == 'temporal_reasoning']
    df_spatial = data[data['category'] == 'spatial_reasoning']
    df_logical = data[data['category'] == 'logical_reasoning']

    score_final = data['score'].mean()
    completion_rate = data['complete'].mean()
    causal_final, causal_comp_rate = df_causal['score'].mean(), df_causal['complete'].mean()
    temporal_final, temporal_comp_rate = df_temporal['score'].mean(), df_temporal['complete'].mean()
    spatial_final, spatial_comp_rate = df_spatial['score'].mean(), df_spatial['complete'].mean()
    logical_final, logical_comp_rate = df_logical['score'].mean(), df_logical['complete'].mean()
    ins_following_average = data['Reasoning'].mean()
    img_consist_average = data['ImageConsistency'].mean()
    generation_quality = data['GenerationQuality'].mean()

    def trans_to_percent(s):
        return 25*(s-1)

    final_score = dict(
        overall=[score_final, trans_to_percent(score_final), completion_rate],
        causal_reasoning=[causal_final, trans_to_percent(causal_final), causal_comp_rate],
        temporal_reasoning=[temporal_final, trans_to_percent(temporal_final), temporal_comp_rate],
        spatial_reasoning=[spatial_final, trans_to_percent(spatial_final), spatial_comp_rate],
        logical_reasoning=[logical_final, trans_to_percent(logical_final), logical_comp_rate],
        Reasoning_total=[ins_following_average, trans_to_percent(ins_following_average), None],
        ImageConsistency_total=[img_consist_average, trans_to_percent(img_consist_average), None],
        GenearationQuality_total=[generation_quality, trans_to_percent(generation_quality), None],
    )

    df = pd.DataFrame(final_score, index=["Score-Origin", "Score-Percentage", "Accuracy"]).T
    df.reset_index(inplace=True)
    df.columns = ["-", "Score-Origin", "Score-Percentage", "Accuracy"]
    df.to_csv(score_file, index=False)


if __name__ == '__main__':
    main()
