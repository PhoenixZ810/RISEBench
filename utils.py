import numpy as np
import base64
import io
import json
import pandas as pd
import pickle
import csv


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret


def prepare_itlist(inputs, image_size=-1,  **kwargs):
    assert np.all([isinstance(x, dict) for x in inputs])
    has_images = np.sum([x['type'] == 'image' for x in inputs])
    if has_images:
        content_list = []
        for msg in inputs:
            if msg['type'] == 'text':
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                from PIL import Image
                img = Image.open(msg['value'])
                b64 = encode_image_to_base64(img, target_size=image_size)
                img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail='high')
                content_list.append(dict(type='image_url', image_url=img_struct))
    else:
        assert all([x['type'] == 'text' for x in inputs])
        text = '\n'.join([x['value'] for x in inputs])
        content_list = [dict(type='text', text=text)]
    return content_list


def prepare_inputs(inputs, system_prompt=None, **kwargs):
    input_msgs = []
    if system_prompt is not None:
        input_msgs.append(dict(role='system', content=system_prompt))
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
    if 'role' in inputs[0]:
        assert inputs[-1]['role'] == 'user', inputs[-1]
        for item in inputs:
            input_msgs.append(dict(role=item['role'], content=prepare_itlist(item['content'], **kwargs)))
    else:
        input_msgs.append(dict(role='user', content=prepare_itlist(inputs, **kwargs)))
    return input_msgs


prompt_consist = """You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) along with a specific modification instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate whether the two images maintain consistency in aspects not related to the given instruction.

## Task
Evaluate the consistency between the images according to the following scale (1 to 5):

- **5 (Perfect Consistency)**: Apart from changes explicitly required by the instruction, all other details (e.g., personal features, clothing, background, layout, colors, positions of objects) are completely identical between the two images.

- **4 (Minor Differences)**: Apart from changes explicitly required by the instruction, the second image is mostly consistent with the original image but contains a minor discrepancy (such as a missing minor personal feature, accessory, or tattoo).

- **3 (Noticeable Differences)**: Apart from changes explicitly required by the instruction, the second image has one significant difference from the original (such as a noticeable alteration in a person's appearance like hair or skin color, or a significant change in background environment).

- **2 (Significant Differences)**: Apart from changes explicitly required by the instruction, the second image has two or more significant differences or multiple noticeable inconsistencies (such as simultaneous changes in both personal appearance and background environment).

- **1 (Severe Differences)**: Apart from changes explicitly required by the instruction, nearly all key details (e.g., gender, major appearance features, background environment, or scene layout) significantly differ from the original image, clearly deviating from the original.

Example:

Original image: A blond, white-skinned man with a tattoo on his right shoulder, furniture in the background.
Instruction: "Show him after gaining fifty pounds."

- **Score 5**: A heavier blond, white-skinned man, tattoo on right shoulder intact, identical furniture and layout.
- **Score 4**: A heavier blond, white-skinned man, missing the tattoo on his right shoulder, identical furniture and layout.
- **Score 3**: A heavier man with black hair instead of blond (change in hair color), or original blond man but with a grassy background instead of furniture.
- **Score 2**: A heavier man with black hair (hair color changed), and the background changed to grass.
- **Score 1**: A heavier black-haired woman, and background changed to grass.

Note: When assigning scores, only consider details unrelated to the instruction. Changes explicitly requested by the instruction should NOT be regarded as inconsistencies.

## Input

**Instruction:** {instruct}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **1-5**"""

prompt_reasoning = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **ground-truth textual description** that represents the intended result of the modification.
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the visual content described in the reference description, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and does not correspond to the description at all.

**Example**
Instruction: Draw what it will look like after it is broken.
Description: An egg is completely broken, with eggshell scattered around and egg white and yolk clearly spilling out.
- **5**: Completely broken egg, clearly scattered eggshells, visible egg white and yolk spilling out.
- **4**: Broken egg, eggshell present but not fully scattered, clearly visible egg white and yolk spilling out.
- **3**: Broken egg with scattered eggshell, but egg white and yolk not spilled or still within eggshell.
- **2**: Only scattered eggshell visible, without clear egg white or yolk.
- **1**: Egg is intact, not broken.

## Input
**Instruction**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_generation = """You are an expert image evaluator. For each task, you will be provided with an **output image** generated by an assistant.

Your task is to independently assess the image along the following dimension and assign an integer score from **1 to 5**:

### Evaluation Dimension: Realism and Generation Quality

Assess the overall visual realism and generation fidelity of the image. Consider the image’s clarity, natural appearance, and compliance with physical plausibility and real-world constraints.

**Scoring Guidelines:**

- **5** The image is sharp, visually coherent, and all elements appear highly realistic and physically plausible.
- **4** The image is clear, with most elements appearing realistic; minor details may show slight unreality.
- **3** The image is mostly clear, but some significant elements appear unrealistic or physically implausible.
- **2** The image is noticeably blurry or contains major unrealistic components or visual distortions.
- **1** The image is extremely blurry, incoherent, or severely unrealistic; realism is nearly absent.

## Output Format

After the evaluation, conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_spatial_ref = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **ground-truth textual description** that represents the intended result of the modification.
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the visual content described in the reference description, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and is entirely unrelated to the description.

## Input
**Instruction**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_qual = """You are a highly skilled image evaluator. Given an image, your task is to assess and determine its clarity and distortion, and then provide a score (an integer between 1 and 5) based on the following criteria:

## Task Requirements:

Determine whether the image has blurriness, distortion, visual defects, or physical inaccuracies.

Assign an appropriate score to the image based on the above criteria, considering its overall quality and detail integrity.

## Scoring Criteria:

- **5 points**: The image is very clear, with complete details, and no noticeable distortion or blurriness. All elements conform to physical laws.
- **4 points**: The image is clear, with only minor blurriness, and no noticeable distortion.
- **3 points**: The image has areas with clarity issues, such as slight blurriness or distortion. Some elements are physically incorrect.
- **2 points**: The image has noticeable blurriness or distortion, with significant detail loss, or lacks physical accuracy.
- **1 point**: The image is severely blurry or distorted, making it difficult to recognize its content, with serious degradation in visual quality, almost unusable.

## Output Format

Provide a clear conclusion with the final score, formatted as follows:

**Final Score:** **1-5**

where X represents the score."""

prompt_spatial_cons = """You are a precise and analytical image consistency evaluator.

You will be given:
- **Image A**: the original image.
- **Image B**: a modified version of Image A.
- **Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your task is to **evaluate how consistent Image B remains with Image A in all aspects *except* those explicitly changed by the instruction.** You must **ignore the instructed changes** and **only assess unintended differences**.

## Evaluation Scale (1 to 5):

- **5** Perfect Consistency
  All elements not related to the instruction are visually identical between Image A and Image B (e.g., style, background, object positions, colors, shapes). No unintended change is present.
- **4** Minor Difference
  One small unintended change is present (e.g., a slight color variation or minor object shape shift), but overall the image remains highly consistent.
- **3** Noticeable Difference
  One major or a few minor unintended changes are present (e.g., an object's shape, color, or background differs noticeably, or style has shifted slightly).
- **2** Significant Inconsistency
  Two or more significant differences unrelated to the instruction (e.g., changes in both object details and background or style), reducing overall fidelity.
- **1** Severe Inconsistency
  Major unintended changes dominate the image (e.g., altered visual style, scene layout, or appearance), clearly breaking consistency with Image A.

> ⚠️ Note:
> - To receive a score of 5, the modified image must be visually identical to the original in every unaffected aspect—symbols, patterns, background, texture, color, category, layout, and style must all match exactly.
> - If the background in the original is vague (e.g., plain white or composed of parts), and the background in Image B is also similar vague, you may disregard background consistency.
> - If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.

## Example

**Original image**: “A silver-framed clock with a white face. Three hands (hour, minute, second) are disassembled and lie beside it.”
**Instruction**: “Assemble the clock to show 9:45.”

**Scoring Criteria:**
- **Score 5**: Frame, face, and hand shapes exactly as original.
- **Score 4**: One hand differs slightly in shape or thickness.
- **Score 3**: All hands identical, differing from original specs, or some other things(like text, furniture in the background) is added.
- **Score 2**: Frame color or face differs, and hand shapes are wrong.
- **Score 1**: Frame, face, and hand appearance all significantly altered, background is totally different.

## Input
**Instruction:** {instruct}

## Output Format
After evaluation, conclude with:

**Final Score:** **1-5**
"""

prompt_logical_txt = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image 1**: The original image.
2. **Image 2**: A generated image from an assistant model.
3. **Problem Description**
4. **Reference Answer**

Your task is to determine whether **Image 2** correctly match the reference answer. Evaluate **Image 2** based on the following metrics, each scored as either 0 or 1:

1. **Logical Correctness (0/1)**
   - Assess whether the content of Image 2 logically matches the reference answer.
   - For example, given Image 1 is a teacher with "1+1=?" on the blackboard, and the problem is "Replace the question mark with the correct answer", if Image 2 replaces the question mark with "2", then the score is 1; other is 0.

2. **Appearance Consistency (0/1)**
   Determine whether the style, environment, arrangement of Image 2 are consistent with Image 1.
   - Consider factors such as color scheme, line/font style, background setting, etc. If Image 2's appearance fully aligns with Image 1, score 1; otherwise, score 0.
   - If the only difference is the actual problem solution (not the style or setting) or slightly lighter/darker color, still assign a score of 1.
   - If Image 2 is created by directly adding a pattern to Image 1, still assign a score of 1.
   - If in Image 1, the nodes and edges form an irregular quadrilateral with varying edge lengths and angles but form a square-like arrangement with equal edge lengths and right angles in Image 2, the score is 0.
## Inputs
**Problem Description**:
{instruct}
**Reference Answer**:
{reference}

## Output
You should provide a step-by-step explanation of how you arrived at each score and conclude with the total scores for all three requirements in the format:

**Final Score**: **X,Y**

where X and Y are the scores for the two metrics (Logical Correctness and Appearance Consistency), respectively.
"""

prompt_logical_img = """**You are a highly skilled image evaluator.** Given a logical problem, you will receive:

1. **Image 1**: A reference ground-truth image that correctly solves the problem.
2. **Image 2**: A generated image from an assistant model.

Your task is to determine whether **Image 2** correctly solves the problem, using **Image 1** as the reference answer. Evaluate **Image 2** based on the following metrics, each scored as either 0 or 1:

1. **Logical Correctness (0/1)**
   Assess whether the content of Image 2 logically equal to Image 1.
   **Examples**
   - In a tic-tac-toe problem, if the positions of the marks in Image 2 are exactly the same as in Image 1, score 1; otherwise, score 0.
   - **If the problem is to , only if Image 2 is completely identical to Image 1(reference answer) in terms of shape, color, arrangement pattern, and pattern orientation, score 1**; otherwise, score 0.
   - If Image 1 only contains 1 gt answer but Image 2 contains several answers, score 0.

2. **Appearance Consistency (0/1)**
   Determine whether the style and environment of Image 2 are consistent with Image 1.
   - Consider factors such as color scheme, line style, background setting, etc. If Image 2's appearance fully aligns with Image 1, score 1; otherwise, score 0.
   - If the only difference is the actual problem solution(such as Image 1 with red line as solution and Image 2 with blue line as solution) or slightly lighter/darker color, still assign a score of 1.
   - If Image 2 is created by directly adding a pattern to Image 1, still assign a score of 1.

**If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.**

## Problem Description
{instruct}

## Output
You should provide a step-by-step explanation of how you arrived at each score and conclude with the total scores for all three requirements in the format:

**Final Score**: **X,Y**

where X and Y are the scores for the two metrics (Logical Correctness and Appearance Consistency), respectively.
"""
