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
        try:
            data.to_excel(f, index=False, engine='xlsxwriter')
        except ModuleNotFoundError:
            data.to_excel(f, index=False)

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


prompt_general_cons = """You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) along with a specific modification instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate whether the two images maintain consistency in aspects not related to the given instruction.

## Task
Evaluate the consistency between the images according to the following scale (1 to 5):

- **5 (Perfect Consistency)**: Apart from changes explicitly required by the instruction, all other meaningful details (e.g., personal features, clothing, background, layout, colors, positions of objects) are preserved. Negligible differences that a human would not readily notice—such as slight shifts in lighting, resolution, compression, or minor texture/color variation—should still be scored 5.

- **4 (Minor Differences)**: Apart from changes explicitly required by the instruction, the second image is mostly consistent with the original image but contains one small yet clearly noticeable discrepancy (such as a missing minor personal feature, accessory, or tattoo).

- **3 (Noticeable Differences)**: Apart from changes explicitly required by the instruction, the second image has one significant difference from the original (such as a noticeable alteration in a person's appearance like hair or skin color, or a significant change in background environment).

- **2 (Significant Differences)**: Apart from changes explicitly required by the instruction, the second image has two or more significant differences or multiple noticeable inconsistencies (such as simultaneous changes in both personal appearance and background environment).

- **1 (Severe Differences)**: Apart from changes explicitly required by the instruction, nearly all key details (e.g., gender, major appearance features, background environment, or scene layout) significantly differ from the original image, clearly deviating from the original.

> ⚠️ Note:
> - When assigning scores, only consider details unrelated to the instruction. Changes explicitly requested by the instruction should NOT be regarded as inconsistencies.
> - Only penalize differences that a human evaluator would readily notice. Subtle or imperceptible differences (e.g., minor changes in lighting, sharpness, compression, fine texture, or slight color/tone shifts) should NOT be penalized.
> - The first image may be a composite stitched from several input images (side-by-side in order) when the instruction asks to combine/merge multiple inputs. In that case, treat every sub-image jointly as the original content, and judge whether the content of each sub-image is faithfully preserved in the second image. Do NOT penalize changes in arrangement, relative position, stitching borders, or seams that result from the required merging/combining.

## Example:

**Original image**: A blond, white-skinned man with a tattoo on his right shoulder, furniture in the background.
**Instruction**: "Show him after gaining fifty pounds."

- **Score 5**: A heavier blond, white-skinned man, tattoo on right shoulder intact, identical furniture and layout.
- **Score 4**: A heavier blond, white-skinned man, missing the tattoo on his right shoulder, identical furniture and layout.
- **Score 3**: A heavier man with black hair instead of blond (change in hair color), or original blond man but with a grassy background instead of furniture.
- **Score 2**: A heavier man with black hair (hair color changed), and the background changed to grass.
- **Score 1**: A heavier black-haired woman, and background changed to grass.

## Input

**Instruction:** {instruct}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **1-5**"""

prompt_general_ref_txt = """You are an expert image evaluator. For each task, you will be provided with:

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
**Instruction:**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_general_ref_img = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **reference image** that represents the intended result of the modification. (Do **not** interpret the reference image as the source/original image before editing. It is only a visual example of the desired final outcome.)
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Output Image and Reference Image
Assess how accurately the output image aligns with the reference image, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the reference image and fully follows the instruction.
- **4**: The image mostly matches the reference image, with only minor discrepancies.
- **3**: The image partially matches the reference image but contains noticeable differences or lacks some details.
- **2**: The image contains clear differences. Important details are missed or inaccurate.
- **1**: The image fails to follow the instruction and is largely unrelated to the reference image.

## Input
**Instruction:**  {instruct}

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_general_ref_txt_w_input = """You are an expert image evaluator. For each task, you will be provided with:

1. An original image.
2. An **instruction** describing how an image should be modified.
3. A **ground-truth textual description** that represents the intended result of the modification.
4. An **output image** generated by an assistant.

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
**Instruction:**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_general_qual = """You are a highly skilled image evaluator. You will receive TWO images:
- **Image 1 (Original)**: the source image before editing.
- **Image 2 (Edited)**: the image produced after an editing instruction was applied to Image 1.

Your task is to assess the **visual plausibility** (clarity and distortion) of **Image 2**, and then provide a score (an integer between 1 and 5) based on the following criteria.

## Task Requirements:

Determine whether **Image 2** has blurriness, distortion, visual defects, or physical inaccuracies, focusing on the regions that were **edited or newly generated**.

Assign an appropriate score based on the following criteria, considering the overall quality and detail integrity of the edited image.

## Scoring Criteria:

- **5 points**: The image is very clear, with complete details, and no noticeable distortion or blurriness. All elements conform to physical laws.
- **4 points**: The image is clear, with only minor blurriness, and no noticeable distortion.
- **3 points**: The image has areas with clarity issues, such as slight blurriness or distortion. Some elements are physically incorrect.
- **2 points**: The image has noticeable blurriness or distortion, with significant detail loss, or lacks physical accuracy.
- **1 point**: The image is severely blurry or distorted, making it difficult to recognize its content, with serious degradation in visual quality, almost unusable.

> ⚠️ Note:
> - Image 1 is provided ONLY as a reference so you can identify which content already existed in the original and which regions were changed. **Focus your judgement on the quality of the edited/newly generated regions.**
> - Do not try to infer the editing instruction; focus only on what was changed or newly generated between the two images.
> - If Image 1 itself is low quality, blurry, or contains defects/imperfections, and these are simply carried over (preserved) into Image 2, do **NOT** penalize Image 2 for them. Only penalize defects that are **introduced by the editing**.
> - Do not penalize the image merely for looking AI-generated or non-photorealistic.

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as follows:

**Final Score:** **1-5**

where X represents the score."""

prompt_counterfactual_qual = """You are an expert image evaluator. You will receive TWO images:
- **Image 1 (Original)**: the source image before editing.
- **Image 2 (Edited)**: the image produced after an editing instruction was applied to Image 1.

The edited image comes from a **counterfactual** task, where the assistant is intentionally asked to depict things that violate common sense or physical/biological laws (e.g., a hand with six fingers, a floating object, an animal with unusual anatomy). Therefore, you must judge **ONLY the perceptual quality** of **Image 2**, and **NOT** its physical plausibility or real-world correctness.

Your task is to assign an integer score from **1 to 5** along the following dimension:

### Evaluation Dimension: Perceptual Quality

Assess only the low-level visual quality of **Image 2** (sharpness, clarity, absence of blur, noise, compression artifacts, broken textures, garbled regions, or rendering glitches), focusing on the regions that were **edited or newly generated**.

**Important:**
- Image 1 is provided ONLY as a reference so you can identify which content already existed in the original and which regions were changed. **Focus your judgement on the quality of the edited/newly generated regions.**
- Do not try to infer the editing instruction; focus only on what was changed or newly generated between the two images.
- If Image 1 itself is low quality, blurry, or contains defects/imperfections, and these are simply carried over (preserved) into Image 2, do **NOT** penalize Image 2 for them. Only penalize defects that are **introduced by the editing**.
- Do **NOT** penalize the image for depicting unrealistic, impossible, or counter-intuitive objects, creatures, or anatomical structures (e.g., extra fingers, extra limbs, impossible arrangements). These are intended by the task.
- Only judge whether the edited image itself is well rendered and visually clean.
- Do not penalize the image for being rendered in 3D or any particular art style.

**Scoring Guidelines:**

- **5** The image is sharp, clean, and visually coherent, with no noticeable blur, noise, or rendering artifacts.
- **4** The image is clear overall; only minor artifacts or slight blur in small regions.
- **3** The image is mostly clear, but some regions show noticeable blur, distortion, or artifacts.
- **2** The image is noticeably blurry or contains major visual artifacts, noise, or garbled regions.
- **1** The image is extremely blurry, noisy, or incoherent, making its content hard to recognize.

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""

prompt_spatial_ref_txt = """You are an expert image evaluator. For each task, you will be provided with:

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
**Instruction:**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_ref_img = """You are an expert image evaluator. For each task, you will be provided with:

1. An **instruction** describing how an image should be modified.
2. A **reference image** that represents the intended result of the modification. (Do **not** interpret the reference image as the source/original image before editing. It is only a visual example of the desired final outcome.)
3. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Output Image and reference Image
Assess how accurately the output image aligns with the reference image, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the reference image and fully follows the instruction.
- **4**: The image mostly matches the reference image, with minor discrepancies.
- **3**: The image partially matches the reference image but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and is entirely unrelated to the reference image.

## Input
**Instruction:**  {instruct}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_ref_txt_w_input = """You are an expert image evaluator. For each task, you will be provided with:

1. An original image.
2. An **instruction** describing how the original image should be modified.
3. A **ground-truth textual description** that represents the intended result of the modification.
4. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the reference description based on the original image, considering the context of the instruction.

**Scoring Criteria:**
- **5**: The image completely matches the description, accurately reflecting every detail and degree.
- **4**: The image mostly matches the description, with minor discrepancies.
- **3**: The image partially matches the description but contains differences or lacks some details.
- **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
- **1**: The image fails to follow the instruction and is entirely unrelated to the description.

## Input
**Instruction:**  {instruct}
**GroundTruth Description:** {reference}

## Output Format

Conclude clearly with the final score, formatted as:

**Final Score:** **X**"""

prompt_spatial_qual = """You are a highly skilled image evaluator. You will receive TWO images:
- **Image 1 (Original)**: the source image before editing.
- **Image 2 (Edited)**: the image produced after an editing instruction was applied to Image 1.

Your task is to assess the **visual plausibility** (clarity and distortion) of **Image 2**, and then provide a score (an integer between 1 and 5) based on the following criteria.

## Task Requirements:

Determine whether **Image 2** has blurriness, distortion, visual defects, or physical inaccuracies, focusing on the regions that were **edited or newly generated**.

Assign an appropriate score based on the following criteria, considering the overall quality and detail integrity of the edited image.

## Scoring Criteria:

- **5 points**: The image is very clear, with complete details, and no noticeable distortion or blurriness. All elements conform to physical laws.
- **4 points**: The image is clear, with only minor blurriness, and no noticeable distortion.
- **3 points**: The image has areas with clarity issues, such as slight blurriness or distortion. Some elements are physically incorrect.
- **2 points**: The image has noticeable blurriness or distortion, with significant detail loss, or lacks physical accuracy.
- **1 point**: The image is severely blurry or distorted, making it difficult to recognize its content, with serious degradation in visual quality, almost unusable.

> ⚠️ Note:
> - Image 1 is provided ONLY as a reference so you can identify which content already existed in the original and which regions were changed. **Focus your judgement on the quality of the edited/newly generated regions.**
> - Do not try to infer the editing instruction; focus only on what was changed or newly generated between the two images.
> - If Image 1 itself is low quality, blurry, or contains defects/imperfections, and these are simply carried over (preserved) into Image 2, do **NOT** penalize Image 2 for them. Only penalize defects that are **introduced by the editing**.
> - Do not penalize the image for being rendered in 3D.

## Output Format

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as follows:

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
  All meaningful elements not related to the instruction are preserved between Image A and Image B (e.g., style, background, object positions, colors, shapes). Negligible differences that a human would not readily notice (e.g., slight changes in lighting, sharpness, compression, fine texture, or minor color/tone shifts) should still be scored 5.
- **4** Minor Difference
  One small but clearly noticeable unintended change is present (e.g., a visible color variation or minor object shape shift), but overall the image remains highly consistent.
- **3** Noticeable Difference
  One major or a few minor unintended changes are present (e.g., an object's shape, color, or background differs noticeably, or style has shifted slightly).
- **2** Significant Inconsistency
  Two or more significant differences unrelated to the instruction (e.g., changes in both object details and background or style), reducing overall fidelity.
- **1** Severe Inconsistency
  Major unintended changes dominate the image (e.g., altered visual style, scene layout, or appearance), clearly breaking consistency with Image A.

> ⚠️ Note:
> - Only penalize differences that a human evaluator would readily notice. Subtle or imperceptible differences in unaffected aspects (lighting, sharpness, compression, fine texture, slight color/tone shifts) should NOT lower the score; in such cases a score of 5 is still appropriate.
> - If the background in the original is vague (e.g., plain white or composed of parts), and the background in Image B is also similar vague, you may disregard background consistency.
> - If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.
> - Image A may be a composite stitched from several input images (side-by-side in order) when the instruction asks to combine/merge multiple inputs. In that case, treat every sub-image in Image A jointly as the original content, and judge whether the content of each sub-image (objects, structure, color, texture, style) is faithfully preserved in Image B. Do NOT penalize changes in arrangement, relative position, stitching borders, or seams that result from the required merging/combining.

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

prompt_logical_cons_ans = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image 1**: The original image.
2. **Image 2**: A generated image from an assistant model.
3. **Problem Description**
4. **Reference Answer**

## Evaluation Task

* Compare Image 2 against Image 1 in terms of visual style, environment, and composition.
* Score each comparison as **1** (consistent) or **0** (inconsistent).

## Consistency Criteria (score 1 if true):

* Color palette, line weight, font/handwriting style, arrangement, and background setting closely match.
* Only the problem's solution differs (e.g., added or removed marks), or color brightness is slightly lighter/darker.
* Image 2 is derived by overlaying a pattern onto Image 1 without altering style or layout.
* **If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.**

> ⚠️ Note:
> - Image 1 may be a composite stitched from several input images (side-by-side in order) when the problem asks to combine/merge multiple inputs. In that case, treat every sub-image jointly as the original content, and judge whether the content of each sub-image is faithfully preserved in the second image. Do NOT penalize changes in arrangement, relative position, stitching borders, or seams that result from the required merging/combining.

## Inconsistency Example (score 0):

* Image 1 shows a quadrilateral with unequal edges and arbitrary angles, but Image 2 depicts a perfect square with equal edges and right angles.

## Inputs
**Problem Description**:
{instruct}
**Reference Answer**:
{reference}

## Output
You should provide a step-by-step explanation of how you arrived at the score and conclude in the format:

**Final Score**: **X**
"""

prompt_logical_cons = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image 1**: The original image.
2. **Image 2**: A generated image answer from an assistant model.
3. **Problem Description**

## Evaluation Task

* Judge the appearance consistency between Image 1 and Image 2.
* Score each comparison as **1** (consistent) or **0** (inconsistent).

## Consistency Criteria (score 1 if true):

* Color palette, line weight, font/handwriting style, arrangement, and background setting closely match.
* Only the problem's solution differs (e.g., added or removed marks), or color brightness is slightly lighter/darker.
* Image 2 is derived by overlaying a pattern onto Image 1 without altering style or layout.
* **If a blue diamond shape appears in the bottom-left corner of Image 2, ignore it; it is a watermark.**

> ⚠️ Note:
> - Image 1 may be a composite stitched from several input images (side-by-side in order) when the problem asks to combine/merge multiple inputs. In that case, treat every sub-image jointly as the original content, and judge whether the content of each sub-image is faithfully preserved in the second image. Do NOT penalize changes in arrangement, relative position, stitching borders, or seams that result from the required merging/combining.

## Inconsistency Example (score 0):

* Image 1 shows a quadrilateral with unequal edges and arbitrary angles, but Image 2 depicts a perfect square with equal edges and right angles.
* Image 2 contains severe blur and distortion.

## Inputs
**Problem Description**:
{instruct}

## Output
Conclude in the format:

**Final Score**: **X**
"""

prompt_logical_ref_txt = """**You are a highly skilled image evaluator.** Given an image with logical problem, you will receive:

1. **Image**: A generated image answer from an assistant model.
2. **Reference Answer**

## Task
Assign a binary score (0 or 1) for **Logical Correctness**:
- **1** if the Generated Image accurately implements the Reference Answer.
- **0** otherwise.

## Requirements
- The original image is not given and do not imagine the original image, just determine whether the generated image matches the reference answer.
- **If the given image contains more content than reference answer, give score 0. For example, reference answer is \"a square\" but image contains more squares or rectangles, give score 0.**

## Inputs
**Reference Answer**:
{reference}

## Output
You should provide a step-by-step explanation of how you arrived at the score and conclude in the format:

**Final Score**: **X**
"""

prompt_logical_ref_img = """**You are a highly skilled image evaluator.** Given a logical problem, you will receive:

1. **Problem Description**
2. **Image 1**: A reference ground-truth image answer that correctly solves the problem. (Do **not** interpret the reference image as the source/original image before editing. It is only a visual example of the desired final outcome.)
3. **Image 2**: A generated image answer from an assistant model.

## Logical Correctness (0/1)
Determine whether the content of *Image 2* solves the problem in the same way as *Image 1* does.

### Scoring rules:

- Score **1** if *Image 2* exactly matches *Image 1* in terms of problem-solving logic and visual outcome.
- Score **0** if there are any differences that affect the correctness of the solution.
**If generated image answer is significantly different from ground-truth, directly score 0.**

### Examples:

- For a tic-tac-toe task, if the positions of all marks in *Image 2* are identical to those in *Image 1*, score 1; otherwise, score 0.
- For a shape-drawing task, *Image 2* must match *Image 1* precisely in shape, color, pattern arrangement, and orientation to receive a score of 1; any deviation results in a score of 0.
- If *Image 1* shows only one correct answer but *Image 2* includes multiple answers, score 0.

## Problem Description
{instruct}

## Output
You should provide a step-by-step explanation of how you arrived at the score and conclude in the format:

**Final Score**: **X**
"""

prompt_logical_ref_txt_w_input = """You are an expert image evaluator. For each task, you will be provided with:

1. An original image containing a logical problem.
2. An **instruction** describing what answer or modification should be drawn on the original image.
3. A **ground-truth textual description** that represents the intended result of the solution.
4. An **output image** generated by an assistant.

Your task is to assess the output image based on the following evaluation dimension:

## Evaluation Dimension: Alignment Between Image and Reference Description
Assess how accurately the output image aligns with the reference description based on the original image, considering the context of the instruction.

**Scoring Criteria:**
- **1**: The output image correctly solves the logical problem and matches the reference description exactly, without adding extra incorrect content.
- **0**: The output image does not correctly solve the problem, does not match the reference description, or contains additional answer content that should not be present.

## Input
**Instruction:**  {instruct}
**GroundTruth Description:** {reference}

Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

**Final Score:** **X**
"""
