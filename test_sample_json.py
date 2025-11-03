import os
os.environ['TRANSFORMERS_CACHE'] = '../cache/'
os.environ['HF_HOME'] = '../cache/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '../cache/'

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from transformers import PreTrainedModel, PreTrainedTokenizer
import random
import numpy as np
import json
from src.eval_data import eval_grade_level, eval_length
from tqdm import tqdm

torch.manual_seed(42)

# MODEL_PATHS = {
#     "length_more": "hallisky/lora-length-long-llama-3-8b",
#     "length_less": "hallisky/lora-length-short-llama-3-8b",
#     "function_more": "hallisky/lora-function-more-llama-3-8b",
#     "function_less": "hallisky/lora-function-less-llama-3-8b",
#     "grade_more": "hallisky/lora-grade-highschool-llama-3-8b", 
#     "grade_less": "hallisky/lora-grade-elementary-llama-3-8b",
#     "formality_more": "hallisky/lora-formality-formal-llama-3-8b", 
#     "formality_less": "hallisky/lora-formality-informal-llama-3-8b", 
#     "sarcasm_more": "hallisky/lora-sarcasm-more-llama-3-8b", 
#     "sarcasm_less": "hallisky/lora-sarcasm-less-llama-3-8b", 
#     "voice_passive": "hallisky/lora-voice-passive-llama-3-8b", 
#     "voice_active": "hallisky/lora-voice-active-llama-3-8b", 
#     "type_persuasive": "hallisky/lora-type-persuasive-llama-3-8b",
#     "type_expository": "hallisky/lora-type-expository-llama-3-8b",
#     "type_narrative": "hallisky/lora-type-narrative-llama-3-8b",
#     "type_descriptive": "hallisky/lora-type-descriptive-llama-3-8b",
# }

# MODEL_PATHS = {
#     "formality_formal": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_formality_formal_4xddp_cuda/complete_epoch_5/",
#     "sarcasm_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_sarcasm_less_4xddp_cuda/complete_epoch_5/",
#     "formality_informal": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_formality_informal_4xddp_cuda/complete_epoch_5/",
#     "sarcasm_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_sarcasm_more_4xddp_cuda/complete_epoch_5/",
#     "function_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_function_less_4xddp_cuda/complete_epoch_5/",
#     "function_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_function_more_4xddp_cuda/complete_epoch_5/",
#     "type_descriptive": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_descriptive_4xddp_cuda/complete_epoch_5/",
#     "grade_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_grade_elementary_4xddp_cuda/complete_epoch_5/",
#     "type_expository": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_expository_4xddp_cuda/complete_epoch_5/",
#     "grade_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_grade_highschool_4xddp_cuda/complete_epoch_5/",
#     "type_narrative": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_narrative_4xddp_cuda/complete_epoch_5/",
#     "length_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_length_long_4xddp_cuda/complete_epoch_5/",
#     "type_persuasive": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_persuasive_4xddp_cuda/complete_epoch_5/",
#     "length_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_length_short_4xddp_cuda/complete_epoch_5/",
#     "voice_active": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_voice_active_4xddp_cuda/complete_epoch_5/",
#     "missspell": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_missspell_4xddp_cuda/complete_epoch_5/",
#     "voice_passive": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_voice_passive_4xddp_cuda/complete_epoch_5/"
# }

# MODEL_PATHS = {
#     "formality_formal": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_formality_formal_4xddp_cuda/complete_epoch_5/",
#     "sarcasm_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_sarcasm_less_4xddp_cuda/complete_epoch_5/",
#     "formality_informal": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_formality_informal_4xddp_cuda/complete_epoch_5/",
#     "sarcasm_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_sarcasm_more_4xddp_cuda/complete_epoch_5/",
#     "function_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_function_less_4xddp_cuda/complete_epoch_5/",
#     "function_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_function_more_4xddp_cuda/complete_epoch_5/",
#     "type_descriptive": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_descriptive_4xddp_cuda/complete_epoch_5/",
#     "grade_less": "/workspace/style_remix/04_07/efficient-transformers/llama_3_1_8B_instruct_grade_elementary_gpu_peft_grade_elem/complete_epoch_5/",
#     "type_expository": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_expository_4xddp_cuda/complete_epoch_5/",
#     "grade_more": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_grade_highschool_4xddp_cuda/complete_epoch_5/",
#     "type_narrative": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_narrative_4xddp_cuda/complete_epoch_5/",
#     "length_more": "/workspace/style_remix/04_07/efficient-transformers/llama_3_1_8B_instruct_length_more_gpu_peft_length_more/complete_epoch_5/",
#     "type_persuasive": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_type_persuasive_4xddp_cuda/complete_epoch_5/",
#     "length_less": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_length_short_4xddp_cuda/complete_epoch_5/",
#     "voice_active": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_voice_active_4xddp_cuda/complete_epoch_5/",
#     "missspell": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_missspell_4xddp_cuda/complete_epoch_5/",
#     "voice_passive": "/workspace/style_remix/efficient-transformers/llama_3_1_8B_voice_passive_4xddp_cuda/complete_epoch_5/"
# }


# MODEL_PATHS = {
#     "formality_formal": "qaic_lora_weights/llama_3_1_8B_formality_formal_4xddp_qaic/complete_epoch_5/",
#     "sarcasm_less": "qaic_lora_weights/llama_3_1_8B_sarcasm_less_4xddp_qaic/complete_epoch_5/",
#     "formality_informal": "qaic_lora_weights/llama_3_1_8B_formality_informal_4xddp_qaic/complete_epoch_5/",
#     "sarcasm_more": "qaic_lora_weights/llama_3_1_8B_sarcasm_more_4xddp_qaic/complete_epoch_5/",
#     "function_less": "qaic_lora_weights/llama_3_1_8B_function_less_4xddp_qaic/complete_epoch_5/",
#     "function_more": "qaic_lora_weights/llama_3_1_8B_function_more_4xddp_qaic/complete_epoch_5/",
#     "type_descriptive": "qaic_lora_weights/llama_3_1_8B_type_descriptive_4xddp_qaic/complete_epoch_5/",
#     "grade_less": "qaic_lora_weights/llama_3_1_8B_grade_elementary_4xddp_qaic/complete_epoch_5/",
#     "type_expository": "qaic_lora_weights/llama_3_1_8B_type_expository_4xddp_qaic/complete_epoch_5/",
#     "grade_more": "qaic_lora_weights/llama_3_1_8B_grade_highschool_4xddp_qaic/complete_epoch_5/",
#     "type_narrative": "qaic_lora_weights/llama_3_1_8B_type_narrative_4xddp_qaic/complete_epoch_5/",
#     "length_more": "qaic_lora_weights/llama_3_1_8B_length_long_4xddp_qaic/complete_epoch_5/",
#     "type_persuasive": "qaic_lora_weights/llama_3_1_8B_type_persuasive_4xddp_qaic/complete_epoch_5/",
#     "length_less": "qaic_lora_weights/llama_3_1_8B_length_short_4xddp_qaic/complete_epoch_5/",
#     "voice_active": "qaic_lora_weights/llama_3_1_8B_voice_active_4xddp_qaic/complete_epoch_5/",
#     "missspell": "qaic_lora_weights/llama_3_1_8B_missspell_4xddp_qaic/complete_epoch_5/",
#     "voice_passive": "qaic_lora_weights/llama_3_1_8B_voice_passive_4xddp_qaic/complete_epoch_5/"
# }

MODEL_PATHS = {
    "formality_formal": "qaic_lora_weights/llama_3_1_8B_formality_formal_4xddp_qaic/complete_epoch_5/",
    "sarcasm_less": "qaic_lora_weights/llama_3_1_8B_sarcasm_less_4xddp_qaic/complete_epoch_5/",
    "formality_informal": "qaic_lora_weights/llama_3_1_8B_formality_informal_4xddp_qaic/complete_epoch_5/",
    "sarcasm_more": "qaic_lora_weights/llama_3_1_8B_sarcasm_more_4xddp_qaic/complete_epoch_5/",
    "function_less": "qaic_lora_weights/llama_3_1_8B_function_less_4xddp_qaic/complete_epoch_5/",
    "function_more": "qaic_lora_weights/llama_3_1_8B_function_more_4xddp_qaic/complete_epoch_5/",
    "type_descriptive": "qaic_lora_weights/llama_3_1_8B_type_descriptive_4xddp_qaic/complete_epoch_5/",
        # "grade_less": "/workspace/StyleRemix/qaic_lora_weights_22_07/llama_3.1_8b_grade_elementary_peft_v2_lr_1e-4_eps_2_21_07_32xddp_sorting_false/complete_epoch_2/",
        "grade_less": "/workspace/StyleRemix/qaic_lora_weights_25_07/llama_3_1_8b_grade_elementary_peft_v3_lr_1e-4_eps_2_25_07_32xddp_sorting_false_v2_linear_w_warmup/complete_epoch_2/",
    "type_persuasive": "qaic_lora_weights/llama_3_1_8B_type_persuasive_4xddp_qaic/complete_epoch_5/",
    "length_more": "qaic_lora_weights/llama_3_1_8B_length_long_4xddp_qaic/complete_epoch_5/",
    "type_persuasive": "qaic_lora_weights/llama_3_1_8B_type_persuasive_4xddp_qaic/complete_epoch_5/",
    "length_less": "qaic_lora_weights/llama_3_1_8B_length_short_4xddp_qaic/complete_epoch_5/",
    "voice_active": "qaic_lora_weights/llama_3_1_8B_voice_active_4xddp_qaic/complete_epoch_5/",
    "missspell": "qaic_lora_weights/llama_3_1_8B_missspell_4xddp_qaic/complete_epoch_5/",
    "voice_passive": "qaic_lora_weights/llama_3_1_8B_voice_passive_4xddp_qaic/complete_epoch_5/",
    "type_expository": "qaic_lora_weights/llama_3_1_8B_type_expository_4xddp_qaic/complete_epoch_5/",
    "grade_more": "qaic_lora_weights/llama_3_1_8B_grade_highschool_4xddp_qaic/complete_epoch_5/",
    "type_narrative": "qaic_lora_weights/llama_3_1_8B_type_narrative_4xddp_qaic/complete_epoch_5/",
        "length_more": "/workspace/StyleRemix/qaic_lora_weights_22_07/llama_3_1_8b_length_more_peft_v2_lr_1e-4_eps_2_21_07_32xddp_sorting_false/complete_epoch_2/",
    "type_persuasive": "qaic_lora_weights/llama_3_1_8B_type_persuasive_4xddp_qaic/complete_epoch_5/",
    "length_less": "qaic_lora_weights/llama_3_1_8B_length_short_4xddp_qaic/complete_epoch_5/",
    "voice_active": "qaic_lora_weights/llama_3_1_8B_voice_active_4xddp_qaic/complete_epoch_5/",
    "missspell": "qaic_lora_weights/llama_3_1_8B_missspell_4xddp_qaic/complete_epoch_5/",
    "voice_passive": "qaic_lora_weights/llama_3_1_8B_voice_passive_4xddp_qaic/complete_epoch_5/"
}

FIRST_MODEL = list(MODEL_PATHS.keys())[0]
# MAX_NEW_TOKENS = 250
MAX_NEW_TOKENS = 1024

# Converts text to the correct format for LoRA adapters in StyleRemix
def convert_data_to_format(text):
    output = f"### Original: {text}\n ### Rewrite:"
    return output


def compute_metric(list_of_text):    
    grade_dict, avg_grade = eval_grade_level(list_of_text)
    # print("Grade: FK: ", grade_dict['fk'][0])
    # print("Grade: LW: ", grade_dict['lw'][0])
    # print("Grade: GF: ", grade_dict['gf'][0])
    # print("Average : ", avg_grade[0])
    _, _, _, words_per_sentence, _ = eval_length(list_of_text)
    # print("Words per sentence: ", words_per_sentence[0])
    return np.mean(avg_grade), np.mean(words_per_sentence)


def remix(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    input_text: str, 
    length: float, 
    function_words: float, 
    grade_level: float, 
    formality: float, 
    sarcasm: float, 
    voice: float, 
    persuasive: float, 
    descriptive: float, 
    narrative: float, 
    expository: float
) -> dict:
    """
    Remix the input text based on various stylistic and structural parameters.

    This function utilizes a pre-trained language model and various LoRA adapters 
    to generate a rewritten version of the input text based on the provided stylistic sliders.

    Args:
        model (PreTrainedModel): The pre-trained language model with LoRA adapters loaded.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the pre-trained model.
        input_text (str): The input text to be remixed.
        length (float): Control the length of the text. Positive for more, negative for less. Range from -1 to 1.
        function_words (float): Control the use of function words. Positive for more, negative for less. Range from -1 to 1.
        grade_level (float): Control the complexity of the text. Positive for more advanced, negative for simpler. Range from -1 to 1.
        formality (float): Control the formality of the text. Positive for more formal, negative for less formal. Range from -1 to 1.
        sarcasm (float): Control the sarcasm of the text. Positive for more sarcastic, negative for less sarcastic. Range from -1 to 1.
        voice (float): Control the voice of the text. Positive for active, negative for passive. Range from -1 to 1.
        persuasive (float): Generate a persuasive style of text. Range from 0 to 1. Only one of `persuasive`, 
                            `descriptive`, `narrative`, or `expository` should be nonzero at a time.
        descriptive (float): Generate a descriptive style of text. Range from 0 to 1. Only one of `persuasive`, 
                             `descriptive`, `narrative`, or `expository` should be nonzero at a time.
        narrative (float): Generate a narrative style of text. Range from 0 to 1. Only one of `persuasive`, 
                           `descriptive`, `narrative`, or `expository` should be nonzero at a time.
        expository (float): Generate an expository style of text. Range from 0 to 1. Only one of `persuasive`, 
                            `descriptive`, `narrative`, or `expository` should be nonzero at a time.

    Returns:
        dict: A dictionary containing the input text, the slider values, the remixed output, 
              and the full model output, including special tokens.
    """

    device = model.device
    
    sliders_dict = {}
    cur_keys = []
    cur_keys.append(("length_more" if length > 0 else (None if length == 0 else "length_less"), abs(length)))
    cur_keys.append(("function_more" if function_words > 0 else (None if function_words == 0 else "function_less"), abs(function_words)))
    cur_keys.append(("grade_more" if grade_level > 0 else (None if grade_level == 0 else "grade_less"), abs(grade_level)))
    cur_keys.append(("sarcasm_more" if sarcasm > 0 else (None if sarcasm == 0 else "sarcasm_less"), abs(sarcasm)))
    cur_keys.append(("formality_more" if formality > 0 else (None if formality == 0 else "formality_less"), abs(formality)))
    cur_keys.append(("voice_active" if voice > 0 else (None if voice == 0 else "voice_passive"),abs(voice)))
    cur_keys.append(("type_persuasive" if persuasive != 0 else None, abs(persuasive)))
    cur_keys.append(("type_descriptive" if descriptive != 0 else None, abs(descriptive)))
    cur_keys.append(("type_narrative" if narrative != 0 else None, abs(narrative)))
    cur_keys.append(("type_expository" if expository != 0 else None, abs(expository)))

    for cur_key in cur_keys:
        if cur_key[0] is not None:
            sliders_dict[cur_key[0]] = cur_key[1]

    # Make the adapter and switch to it
    print(f"Slider dict: {sliders_dict}")
    if len(sliders_dict) > 0:
        combo_adapter_name = ""
        for slider_key in sliders_dict:
            combo_adapter_name += slider_key + str(int(100*sliders_dict[slider_key])) + "-"
        combo_adapter_name = combo_adapter_name[:-1]

        # Add and set the weighted adapater
        model.add_weighted_adapter(
            list(sliders_dict.keys()),
            weights = list(sliders_dict.values()),
            adapter_name = combo_adapter_name,
            combination_type = "cat"
        )
        model.set_adapter(combo_adapter_name)
        
        # Convert the list of strings in data to a list of model inputs
        converted_text = convert_data_to_format(input_text)
        inputs = tokenizer(converted_text, return_tensors="pt", max_length=2048, truncation=True).to(device)
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad(): 
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_p = 0.95)
            # outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, stopping_criteria=None)
        response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    else:
        response = input_text # If no sliders passed, do not do anything
        full_output = response
        # NOTE: Below code is useful when we want to ingest the text generated via Imagine UI and compute metrics on it
        # Convert the list of strings in data to a list of model inputs
        # converted_text = convert_data_to_format(input_text)
        # inputs = tokenizer(converted_text, return_tensors="pt", max_length=2048, truncation=True).to(device)
        # input_length = inputs.input_ids.shape[1]
        # with torch.no_grad(): 
        #     outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_p = 0.95)
        #     # outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, stopping_criteria=None)
        # response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
        # full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    latest_obfuscation = {
        "input_text": input_text,
        "sliders": {
            "length": length,
            "function_words": function_words,
            "grade_level": grade_level,
            "sarcasm": sarcasm,
            "formality": formality,
            "voice": voice,
            "persuasive": persuasive,
            "descriptive": descriptive,
            "narrative": narrative,
            "expository": expository
        },
        "input": input_text,
        "output": response,
        "full_output": full_output
    }
    return latest_obfuscation

def main(args):
    with open(args.json_file, "r") as f:
        text_data = json.load(f)

    texts = []
    for key, value in text_data.items():
        value = value.replace("### Original: ", "")
        texts.append(value)
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, add_eos_token=False, padding_side="left")
    tokenizer.add_special_tokens({'pad_token': '<padding_token>'})
    base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device) # device_map="auto" requires accelerate
    base_model.resize_token_embeddings(len(tokenizer)) # Resize to add pad token. Value doesn't matter
    
    # Load in the first LoRA adapter
    model = PeftModel.from_pretrained(base_model, MODEL_PATHS[FIRST_MODEL], adapter_name=FIRST_MODEL).to(device)
    for cur_adapter in MODEL_PATHS.keys(): # Load in the rest of the models
        if cur_adapter != FIRST_MODEL:
            model.load_adapter(MODEL_PATHS[cur_adapter], adapter_name=cur_adapter)
    model.to(device)
    model.eval()

    comb_output_txt = []
    result = {}
    for i, t in enumerate(tqdm(texts)):
        if not args.random_weights:
            cur_remix = remix(
                model,
                tokenizer,
                t,
                args.length,
                args.function_words,
                args.grade_level,
                args.formality,
                args.sarcasm,
                args.voice,
                args.persuasive,
                args.descriptive,
                args.narrative,
                args.expository
            )
        else:
            categories = ["length", "function_words", "grade_level", "formality", "sarcasm", "voice", "writing_style"]
            
            # Initialize dict of slider values to 0 for all elements in categories, except for "writing style", which 
            # should be substituted with persuasive, ..., expository
            style_elements_weights = {c: 0 for c in categories if c != "writing_style"}
            for c in ["persuasive", "descriptive", "narrative", "expository"]:
                style_elements_weights[c] = 0
            
            # Sample args.num_random adapters from categories
            chosen_categories = random.sample(categories, args.num_random)

            # For each of the chosen vectors, randomly select a weight sampled from a normal distribution with mean 
            # of args.random_weight_mean and std of args.random_weight_std and set value in the dict.
            for category in chosen_categories:
                weight = min(np.random.normal(args.random_weight_mean, args.random_weight_std), 1)
                if category == "writing_style":
                    chosen_style = random.choice(["persuasive", "descriptive", "narrative", "expository"])
                    style_elements_weights[chosen_style] = weight
                else:
                    # Make weight negative and positive with 50% probability each
                    style_elements_weights[category] = weight if random.random() < 0.5 else -weight

            print("Random style element weights chosen:")
            for c in style_elements_weights:
                if style_elements_weights[c] != 0:
                    print(c, style_elements_weights[c])

            cur_remix = remix(
                model,
                tokenizer,
                t,
                **style_elements_weights
            )
            
        print(f"Input: {t}\nOutput: {cur_remix['output']}\n\n")
        result[i] = {}
        result[i]['input_txt'] = t
        result[i]['output_txt'] = cur_remix['output']
        comb_output_txt.append(result[i]['output_txt'])
    
    # NOTE: Below code is useful when we want to ingest the text generated via Imagine UI and compute metrics on it
    # imagine_ui_output = "/workspace/StyleRemix/imaging_ui_response.txt"
    # with open(imagine_ui_output, "r") as f:
    #     lines = f.readlines()
        
    # comb_output_txt = []
    # for line in lines:
    #     comb_output_txt.append(line.strip())

    avg_grade, avg_words_per_sentence = compute_metric(comb_output_txt)
    print(f"Average for grade-level (elementary): {avg_grade:.4f}")
    print(f"Avg. words per sentence: {avg_words_per_sentence:.4f}")
    
    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=4)
    print("Results saved in ", args.result_file)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Script to generate with vLLM')

    # Add arguments for sliders ranging from -1 to 1 as floats
    parser.add_argument('--length', type=float, default=0, help='Control the length of the text. Positive for more, negative for less, range from -1 to 1.')
    parser.add_argument('--function_words', type=float, default=0, help='Control the use of function words. Positive for more, negative for less, range from -1 to 1.')
    parser.add_argument('--grade_level', type=float, default=0, help='Control the complexity of the text. Positive for more advanced, negative for simpler, range from -1 to 1.')
    parser.add_argument('--formality', type=float, default=0, help='Control the formality of the text. Positive for more formal, negative for less formal, range from -1 to 1.')
    parser.add_argument('--sarcasm', type=float, default=0., help='Control the sarcasm of the text. Positive for more sarcastic, negative for less sarcastic, range from -1 to 1.')
    parser.add_argument('--voice', type=float, default=0, help='Control the voice of the text. Positive for active, negative for passive, range from -1 to 1.')
    parser.add_argument('--json_file', type=str, help='Json file containing input text.')
    parser.add_argument('--result_file', type=str, help='Result file name.')

    # Add mutually exclusive group for text type with float range from 0 to 1
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument('--persuasive', type=float, default=0, help='Generate a persuasive style of text. Range from 0 to 1.')
    type_group.add_argument('--descriptive', type=float, default=0, help='Generate a descriptive style of text. Range from 0 to 1.')
    type_group.add_argument('--narrative', type=float, default=0, help='Generate a narrative style of text. Range from 0 to 1.')
    type_group.add_argument('--expository', type=float, default=0, help='Generate an expository style of text. Range from 0 to 1.')

    # Arguments for random 
    parser.add_argument("--random_weights", action="store_true", help="Randomly assign weights to style axes")
    parser.add_argument("--num_random", type=int, default=3, help="Number of random style axes to use")
    parser.add_argument("--random_weight_mean", type=int, default=0.8, help="Mean of random feature values selected")
    parser.add_argument("--random_weight_std", type=int, default=0.05, help="Standard deviation of random feature values selected")

    # Parse arguments
    main(parser.parse_args())

    """
    Example commands to run StyleRemix on text:

    # For baseline
    python3 test_sample_json.py --length 0.0 --json_file prompts_long_set.json --result_file results_baseline_04_07.json
    
    # For grade elementary lora
    python3 test_sample_json.py --grade_level -1.0 --json_file prompts_long_set.json --result_file results_grade_level_elem_21_07.json

    # For sentence lengthening
    python3 test_sample_json.py --length 1.0 --json_file prompts_long_set.json --result_file results_sentence_long_21_07.json

    # For both
    python3 test_sample_json.py --length 1.0 --grade_level -1.0 --json_file prompts_long_set.json --result_file results_grade_level_elem_sentence_long_21_07.json

    """
