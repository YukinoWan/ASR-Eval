import json
import re

# filter llm generations to get the output
def filter(output):
    cleaned_text= output.strip("\n").split("\n")[0].strip().strip(".").strip().lower()
    if "[" in cleaned_text and "]" in cleaned_text:
        try:
            cleaned_text = json.loads(cleaned_text)[0]
        except:
            cleaned_text = cleaned_text.replace("['", "").replace("']", "").strip().strip(".").strip()
    else:
        cleaned_text = cleaned_text.replace("['", "").replace("']", "").strip().strip(".").strip()
    cleaned_text = cleaned_text.replace("\"", "").replace("\"", "").strip().strip(".").strip()

    return cleaned_text

def mt_filter(output):
    if isinstance(output, list):
        output = output[0]
    try:
        outputs = output.split("\n\n")
    except:
        print("PROBLEM!!!!!", output)
        print(type(output))
        assert False
    if "translation" in outputs[0]:
        print("PROBLEM!!!!!", output)
        try:
            if "translation" in outputs[1]:
                print("PROBLEM!!!!!", output)
                try:
                    clean_output = outputs[2]
                except:
                    clean_output = outputs[1]
            else:
                clean_output = outputs[1]
        except:
            clean_output = outputs[0]
    else:
        clean_output = outputs[0]

    return clean_output

def st_filter(output):
    if isinstance(output, list):
        output = output[0]
    try:
        outputs = output.split("\n\n")
    except:
        print("PROBLEM!!!!!", output)
        print(type(output))
        assert False
    if "\"" in outputs[0] or "'" in outputs[0]:
        input_string = outputs[0]
    else:
        input_string = outputs[1]
    match = re.search(r'"(.*?)"', input_string)
    if match:
        extracted_sentence = match.group(1)
    else:
        match = re.search(r"\['(.*?)'\]", input_string)
        if match:
            extracted_sentence = match.group(1)
        else:
            match = re.search(r"'(.*?)'", input_string)
            if match:
                extracted_sentence = match.group(1)
            else:
                print("PROBLEM!!!", outputs[0])
                try:
                    extracted_sentence = outputs[0].split(": ")[0]
                except:
                    assert False
    clean_output = extracted_sentence
    return clean_output



