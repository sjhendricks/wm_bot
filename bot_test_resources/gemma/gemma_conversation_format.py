import json

SYSTEM_PROMPT = "You are a helpful William and Mary Advisor. Be concise and professional."

def convert_example_for_gemma(example):
    new_convos = []
    # We grab the conversations list
    old_convos = example.get("conversations", [])
    
    for i, msg in enumerate(old_convos):
        role = msg.get("from")
        content = msg.get("value", "").strip()

        if role == "human":
            role = "user"
            # IF THIS IS THE FIRST MESSAGE: Fold the system prompt in
            if i == 0:
                content = f"{SYSTEM_PROMPT}\n\nUser Question: {content}"
            
            # Use your existing cleaning logic here
            if "\n\n" in content and i != 0: # Only split if it's not the first (already modified) message
                 content = content.split("\n\n", 1)[-1].strip()

        elif role == "gpt":
            role = "assistant"

        new_convos.append({
            "role": role,
            "content": content
        })

    return {"messages": new_convos}



def convert_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)
            converted = convert_example_for_gemma(data)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_jsonl("data/fine_tuning/fine_tuning_data.jsonl", "bot_test_resources/gemma/gemma_formatted_fine_tuning_data.jsonl")
    print("Done converting JSONL with system prompt and cleaning!")