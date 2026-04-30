import json

SYSTEM_PROMPT = "You are a professional William & Mary Academic Advisor. Your sole purpose is to assist students with W&M-related inquiries. If a student asks a question that is unrelated to William & Mary, you must politely decline to answer and offer to help them with their academic journey instead. Do not provide general knowledge or instructions outside of this scope."

def convert_example(example):
    new_convos = []

    # Add system message first
    new_convos.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    for msg in example.get("conversations", []):
        role = msg.get("from")
        content = msg.get("value")

        # Map roles and clean content
        if role == "human":
            role = "user"
            
            # --- NEW CLEANING LOGIC ---
            # If a double newline exists, split once and keep everything after it.
            # We also add .strip() to remove any lingering spaces.
            if "\n\n" in content:
                content = content.split("\n\n", 1)[-1].strip()
            else:
                content = content.strip()
                
        elif role == "gpt":
            role = "assistant"
            # Good practice to strip trailing spaces from the model's target response too
            content = content.strip()

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
            converted = convert_example(data)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_jsonl("data/fine_tuning/fine_tuning_data.jsonl", "data/fine_tuning/formatted_fine_tuning_data.jsonl")
    print("Done converting JSONL with system prompt and cleaning!")