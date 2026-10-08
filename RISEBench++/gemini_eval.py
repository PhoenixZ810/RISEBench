import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable
import time
import re
from utils import *
from openai import OpenAI

# Run from RISEBench++ after configuring OPENAI_API_KEY and OPENAI_BASE_URL:
# python gemini_eval.py --data data/overall_data.json --input data \
#     --output outputs/MODEL_NAME --model MODEL_NAME --nproc 10

client = None


def create_client():
    """Configure the OpenAI-compatible endpoint used by the Gemini judges."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    api_base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running evaluation.")
    if not api_base:
        raise ValueError("Set OPENAI_BASE_URL to your OpenAI-compatible Gemini endpoint.")
    return OpenAI(api_key=api_key, base_url=api_base)


CATEGORY_DISPLAY_NAMES = {
    'temporal_reasoning': 'Temporal',
    'causal_reasoning': 'Causal',
    'spatial_reasoning': 'Spatial',
    'logical_reasoning': 'Logical',
    'counterfactual_reasoning': 'Counterfactual',
    'hybrid_reasoning': 'Hybrid',
}

def gpt_generate(inputs, model='google/gemini-3-flash-preview', temperature=0, max_tokens=4096, image_size=768, **kwargs):
    input_msgs = prepare_inputs(inputs, image_size=image_size)
    temperature = kwargs.pop('temperature', temperature)
    max_tokens = kwargs.pop('max_tokens', max_tokens)

    retries=5
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=input_msgs,
                # max_tokens=max_tokens,
                # temperature=temperature,
                # **kwargs
            )
            break
        except Exception as e:
            print(f"❌ [Attempt {attempt}/{retries}] Unexpected error: {e}")
            if attempt==retries:
                raise e
            time.sleep(3)

    ret_code = 0
    try:
        answer = response.choices[0].message.content.strip()
        return ret_code, answer, response
    except Exception as err:
        raise RuntimeError("The judge returned an invalid or empty text response.") from err

    

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
                    exception = futures[idx].exception()
                    if exception is not None:
                        raise exception
                    else:
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


def build_min_score_judgement(index, category, missing_path, dimension, score=1):
    return (
        f"[Auto fallback] Missing output image for sample {index} ({category}): {missing_path}\n"
        f"{dimension} is assigned the minimum score because the model output image is missing.\n"
        f"Final Score: {score}"
    )


def format_instruction(instruct):
    if isinstance(instruct, (list, tuple)):
        return '\n'.join([f'{idx + 1}. {x}' for idx, x in enumerate(instruct)])
    return instruct


def eval_vanilla(item, input_dir, output_dir, **kwargs):
    instruct = format_instruction(item['instruction'])
    index = item['index']
    category = item['category']
    is_multi_turn = 'multi_turn' in item and not pd.isna(item['multi_turn'])
    category_list = category if isinstance(category, (list, tuple)) else [category]
    is_logical_hybrid = is_multi_turn and 'logical_reasoning' in category_list
    output_category = 'hybrid_reasoning' if is_multi_turn else category
    output_dir = osp.join(output_dir, f'images/{output_category}')
    expected_img = osp.join(output_dir, f"{index}.[png|jpg|jpeg]")
    try:
        img2 = find_image(output_dir, index)
    except FileNotFoundError:
        print(f"Missing output image for sample {index} in {output_dir}. Assigning minimum score.")
        consistency_free = 'consistency_free' in item and not pd.isna(item['consistency_free'])
        fallback_reasoning = build_min_score_judgement(index, output_category, expected_img, 'Reasoning')

        if category == 'logical_reasoning' or is_logical_hybrid:
            return dict(
                judge1=build_min_score_judgement(index, output_category, expected_img, 'Consistency', score=0),
                judge2=build_min_score_judgement(index, output_category, expected_img, 'Reasoning', score=0),
            )

        fallback_quality = build_min_score_judgement(index, output_category, expected_img, 'Visual plausibility')
        if consistency_free:
            return dict(judge1=None, judge2=fallback_reasoning, judge3=fallback_quality)

        return dict(
            judge1=build_min_score_judgement(index, output_category, expected_img, 'Consistency'),
            judge2=fallback_reasoning,
            judge3=fallback_quality,
        )
    judge_exist = item.get('judge', None)
    judge_rea_require_img = False
    
    if (is_multi_turn and not is_logical_hybrid) or category in ['temporal_reasoning', 'causal_reasoning']:
        img1 = osp.join(input_dir, item['image'][0])
        if "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img = True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_rea = prompt_general_ref_img.format(instruct=instruct)
        elif "reasoning_img" in item and not pd.isna(item['reasoning_img']):
            judge_rea_require_img = True
            reference = item['reference_txt']
            prompt_rea = prompt_general_ref_txt_w_input.format(instruct=instruct, reference=reference)
        else:
            reference = item['reference_txt']
            prompt_rea = prompt_general_ref_txt.format(instruct=instruct, reference=reference)

        prompt_cons = prompt_general_cons.format(instruct=instruct)
        prompt_qua = prompt_counterfactual_qual if 'counterfactual_reasoning' in category_list else prompt_general_qual

    elif category == 'spatial_reasoning':
        img1 = osp.join(input_dir, item['image'][0])
        if "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img = True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_rea = prompt_spatial_ref_img.format(instruct=instruct)
        elif "reasoning_img" in item and not pd.isna(item['reasoning_img']):
            judge_rea_require_img = True
            reference = item['reference_txt']
            prompt_rea = prompt_spatial_ref_txt_w_input.format(instruct=instruct, reference=reference)
        else:
            reference = item['reference_txt']
            prompt_rea = prompt_spatial_ref_txt.format(instruct=instruct, reference=reference)

        prompt_cons = prompt_spatial_cons.format(instruct=instruct)
        prompt_qua = prompt_spatial_qual

    elif category == 'logical_reasoning' or is_logical_hybrid:
        img1 = osp.join(input_dir, item['image'][0])
        if "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img=True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_cons = prompt_logical_cons.format(instruct=instruct)
            prompt_rea = prompt_logical_ref_img.format(instruct=instruct)
        elif "reasoning_img" in item and not pd.isna(item['reasoning_img']):
            judge_rea_require_img=True
            reference = item['reference_txt']
            prompt_cons = prompt_logical_cons_ans.format(instruct=instruct, reference=reference)
            prompt_rea = prompt_logical_ref_txt_w_input.format(instruct=instruct, reference=reference)
        else:
            reference = item['reference_txt']
            prompt_cons = prompt_logical_cons_ans.format(instruct=instruct, reference=reference)
            prompt_rea = prompt_logical_ref_txt.format(reference=reference)

    elif category == 'counterfactual_reasoning':
        img1 = osp.join(input_dir, item['image'][0])
        if "reasoning_img" in item and not pd.isna(item['reasoning_img']):
            judge_rea_require_img=True
            reference = item['reference_txt']
            prompt_rea = prompt_general_ref_txt_w_input.format(instruct=instruct, reference=reference)
        else:
            reference = item['reference_txt']
            prompt_rea = prompt_general_ref_txt.format(instruct=instruct, reference=reference)
        
        prompt_cons = prompt_general_cons.format(instruct=instruct)
        prompt_qua = prompt_counterfactual_qual

    if 'consistency_free' in item and not pd.isna(item['consistency_free']):
        consist_judge = None
        print('Consistency Judgement not required. Ignore.')
    else:
        if judge_exist and 'judge1' in judge_exist:
            consist_judge = judge_exist['judge1']
        else:
            if 'consistency_img' in item and not pd.isna(item['consistency_img']):
                cons_img = osp.join(input_dir, item['consistency_img'])
            else:
                cons_img = osp.join(input_dir, item['image'][0])
            message = []
            text = {'type': 'text', 'value': prompt_cons}
            image1 = {
                'type': 'image',
                'value': cons_img,
            }
            image2 = {
                'type': 'image',
                'value': img2,
            }
            message.append(text)
            message.append(image1)
            message.append(image2)

            ret_code, consist_judge, response = gpt_generate(message, **kwargs)

    if judge_exist and 'judge2' in judge_exist:
        answer2 = judge_exist['judge2']
    else:
        if judge_rea_require_img:
            message2 = [
                {'type': 'text', 'value': prompt_rea}, 
                {'type': 'image','value': img1},
                {'type': 'image','value': img2}
                ]
        else:
            message2 = [{'type': 'text', 'value': prompt_rea}, {
                'type': 'image',
                'value': img2,
            }]

        ret_code2, answer2, response2 = gpt_generate(message2, **kwargs)

    if (is_multi_turn and not is_logical_hybrid) or category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning', 'counterfactual_reasoning']:
        if judge_exist and 'judge3' in judge_exist:
            answer3 = judge_exist['judge3']
        else:
            if 'consistency_img' in item and not pd.isna(item['consistency_img']):
                qual_src_img = osp.join(input_dir, item['consistency_img'])
            else:
                qual_src_img = osp.join(input_dir, item['image'][0])
            message3 = [
                {'type': 'text', 'value': prompt_qua},
                {'type': 'image', 'value': qual_src_img},
                {'type': 'image', 'value': img2},
            ]

            ret_code3, answer3, response3 = gpt_generate(message3, model='google/gemini-3.1-flash-lite', **kwargs)

        return dict(judge1=consist_judge, judge2=answer2, judge3=answer3)
    else:
        return dict(judge1=consist_judge, judge2=answer2)


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
    consistency_free = 'consistency_free' in row and not pd.isna(row['consistency_free'])
    category = row['category']
    is_multi_turn = 'multi_turn' in row and not pd.isna(row['multi_turn'])
    category_list = category if isinstance(category, (list, tuple)) else [category]
    is_logical_hybrid = is_multi_turn and 'logical_reasoning' in category_list

    if (is_multi_turn and not is_logical_hybrid) or category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning', 'counterfactual_reasoning']:
        if consistency_free:
            score = 0.2 * row['VisualPlausibility'] + 0.8 * row['Reasoning']
        else:
            score = 0.3 * row['ApprConsistency'] + 0.5 * row['Reasoning'] + 0.2 * row['VisualPlausibility']
        
    elif category == 'logical_reasoning' or is_logical_hybrid:
        score = 0.3 * row['ApprConsistency'] + 0.7 * row['Reasoning']
    else:
        return None

    if row['Reasoning'] == 1:
        score = score * 0.5
        score = 1 if score<1 else score

    return score

def calculate_completion(row):
    consistency_free = 'consistency_free' in row and not pd.isna(row['consistency_free'])
    category = row['category']
    is_multi_turn = 'multi_turn' in row and not pd.isna(row['multi_turn'])
    category_list = category if isinstance(category, (list, tuple)) else [category]
    is_logical_hybrid = is_multi_turn and 'logical_reasoning' in category_list

    if (is_multi_turn and not is_logical_hybrid) or category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning', 'counterfactual_reasoning']:
        if consistency_free:
            return 1 if row['Reasoning'] == 5 and row['VisualPlausibility'] == 5 else 0
        return 1 if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 and row['VisualPlausibility'] == 5 else 0

    elif category=='logical_reasoning' or is_logical_hybrid:
        return 1 if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 else 0

    return None


def trans_to_percent(score):
    if pd.isna(score):
        return None
    return 25 * (score - 1)


def append_summary_row(summary_rows, label, score, accuracy):
    summary_rows.append({
        '-': label,
        'Score-Origin': score,
        'Score-Percentage': trans_to_percent(score),
        'Accuracy': accuracy,
    })


def append_group_summaries(summary_rows, data, group_cols, label_builder):
    grouped = (
        data.groupby(group_cols, dropna=False)
        .agg(score=('score', 'mean'), accuracy=('complete', 'mean'))
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        append_summary_row(summary_rows, label_builder(row), row.score, row.accuracy)

def main():
    parser = argparse.ArgumentParser(description='Evaluate RISEBench++ model outputs with Gemini judges.')
    parser.add_argument('--data', type=str, required=True, help='Json Path')
    parser.add_argument('--output', type=str, required=True, help='Output Image Dir, outputs/MODEL_NAME')
    parser.add_argument('--input', type=str, default='data', help='Input Image Dir')
    parser.add_argument('--prefix', type=str, default=None, help='output json prefix')
    parser.add_argument('--model', type=str, default=None, help='Name of the model being evaluated (used in result filenames)')
    parser.add_argument('--nproc', type=int, default=10, help='Number of concurrent API workers (default: 10)')

    args = parser.parse_args()

    global client
    try:
        client = create_client()
    except ValueError as exc:
        parser.error(str(exc))

    model_name = osp.basename(osp.normpath(args.output)) if args.model is None else args.model
    if not args.prefix:
        tmp_file = f"{args.output}/{model_name}.pkl"
        judge_res = f"{args.output}/{model_name}_judge.xlsx"
        score_file = f"{args.output}/{model_name}_judge.csv"
    else:
        tmp_file = f"{args.output}/{args.prefix}_{model_name}.pkl"
        judge_res = f"{args.output}/{args.prefix}_{model_name}_judge.xlsx"
        score_file = f"{args.output}/{args.prefix}_{model_name}_judge.csv"

    os.makedirs(args.output, exist_ok=True)

    with open(args.data, encoding="utf-8") as data_file:
        data = json.load(data_file)
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
    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=args.nproc, chunksize=args.nproc, save=tmp_file, keys=keys)
        result = load(tmp_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v

    judges = [result[i] for i in data['index']]

    scores, judge_combine, judge_cons, judge_reas, judge_qua = [], [], [], [], []

    for judge in judges:
        if judge['judge1'] is None:
            judge_combine.append(
                'REASONING\n\n'
                + judge['judge2']
                + '\n\nQUALITY\n\n'
                + judge['judge3']
            )
            judge_cons.append(None)
            judge_reas.append(judge['judge2'])
            judge_qua.append(judge['judge3'])

            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score2 or not score3:
                score=None
            else:
                score = [None]+score2+score3
        elif 'judge3' not in judge:
            judge_combine.append(
                'CONSISTENCY\n\n'
                + judge['judge1']
                + '\n\nREASONING\n\n'
                + judge['judge2']
            )
            judge_cons.append(judge['judge1'])
            judge_reas.append(judge['judge2'])
            judge_qua.append(None)

            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2
        elif 'judge2' not in judge:
            judge_combine.append(judge['judge1'])
            score = [extract(judge['judge1'])[1], extract(judge['judge1'])[0]]
        else:
            try:
                judge_combine.append(
                    'CONSISTENCY\n\n'
                    + judge['judge1']
                    + '\n\nREASONING\n\n'
                    + judge['judge2']
                    + '\n\nQUALITY\n\n'
                    + judge['judge3']
                )
                judge_cons.append(judge['judge1'])
                judge_reas.append(judge['judge2'])
                judge_qua.append(judge['judge3'])
            except Exception as e:
                raise ValueError("Malformed cached judgement: expected text for all dimensions.") from e
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score1 or not score2 or not score3:
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
    data['Reasoning'] = reasoning
    data['ApprConsistency'] = img_consist
    data['VisualPlausibility'] = gen_quality
    data['match_log'] = match_log
    data['judge_cons'] = judge_cons
    data['judge_reas'] = judge_reas
    data['judge_qua'] = judge_qua

    data['score'] = data.apply(calculate_score, axis=1)
    data['complete'] = data.apply(calculate_completion, axis=1)

    dump(data, judge_res)

    summary_rows = []
    append_summary_row(summary_rows, 'Overall', data['score'].mean(), data['complete'].mean())
    append_summary_row(summary_rows, 'Overall_Reasoning', data['Reasoning'].mean(), None)
    append_summary_row(summary_rows, 'Overall_ApprConsistency', data['ApprConsistency'].mean(), None)
    append_summary_row(summary_rows, 'Overall_VisualPlausibility', data['VisualPlausibility'].mean(), None)

    hybrid_mask = data.apply(lambda row: 'multi_turn' in row and not pd.isna(row['multi_turn']), axis=1)
    normal_data = data[~hybrid_mask]
    hybrid_data = data[hybrid_mask]

    for category in normal_data['category'].dropna().drop_duplicates():
        category_df = normal_data[normal_data['category'] == category]
        category_label = CATEGORY_DISPLAY_NAMES.get(category, category)
        append_summary_row(summary_rows, category_label, category_df['score'].mean(), category_df['complete'].mean())
        append_summary_row(summary_rows, f'{category_label}_Reasoning', category_df['Reasoning'].mean(), None)
        append_summary_row(summary_rows, f'{category_label}_Consistency', category_df['ApprConsistency'].mean(), None)
        if category != 'logical_reasoning':
            append_summary_row(summary_rows, f'{category_label}_Quality', category_df['VisualPlausibility'].mean(), None)

    if len(hybrid_data):
        append_summary_row(summary_rows, 'Hybrid', hybrid_data['score'].mean(), hybrid_data['complete'].mean())
        append_summary_row(summary_rows, 'Hybrid_Reasoning', hybrid_data['Reasoning'].mean(), None)
        append_summary_row(summary_rows, 'Hybrid_ApprConsistency', hybrid_data['ApprConsistency'].mean(), None)
        append_summary_row(summary_rows, 'Hybrid_Quality', hybrid_data['VisualPlausibility'].mean(), None)

    append_group_summaries(
        summary_rows,
        normal_data,
        ['category', 'subcategory'],
        lambda row: f"subcategory/{row.category}/{row.subcategory}"
    )
    append_group_summaries(
        summary_rows,
        normal_data,
        ['category', 'subcategory', 'task_type'],
        lambda row: f"task_type/{row.category}/{row.subcategory}/{row.task_type}"
    )

    df = pd.DataFrame(summary_rows, columns=["-", "Score-Origin", "Score-Percentage", "Accuracy"])
    df.to_csv(score_file, index=False)


if __name__ == '__main__':
    main()
