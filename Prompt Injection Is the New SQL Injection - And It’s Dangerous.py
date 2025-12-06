from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(

    model_name,
    dtype= "auto",
    device_map = "cuda",
    load_in_4bit = True
)

SYSTEM_PROMPTS = """"
You are a computer science tutor chatbot. Your rules:

1. You must ONLY answer questions about computer science topics including:
   - Programming languages (Python, Java, C++, etc.)
   - Algorithms and data structures
   - Databases and SQL
   - Operating systems
   - Computer networks
   - Software engineering
   - Cybersecurity basics
   - Computer architecture
   - Web development

2. You must REFUSE to answer any questions that are not related to computer science.

3. If a user asks about non-CS topics, politely say: "I can only answer computer science questions."

4. Keep your answers educational, accurate, and helpful.

5. Always prioritize clarity and provide examples when helpful.

Remember: You are strictly a computer science tutor and nothing else.
"""


def chat_with_cs_chatbot():
    print("COMPUTER SCIENCE TUTOR CHATBOT")
    print("=")
    print("This chatbot only answers computer science questions.")
    print("Type 'exit' to end the conversation.\n")


    conversation_history = []
    while True:
        user_input = input("\n You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodby see you soon \n")
            break
        if not user_input:
            print("Ask a Question in Computer Science.")
            continue
        full_prompt = f"{SYSTEM_PROMPTS} \n\n User Question {user_input}\ Assistant:"
        try:
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            generate_ids = model.generate(
                **inputs,
                max_new_tokens= 1024,
                temperature = 0.7,
                do_sample = True,
                pad_token_id = tokenizer.eos_token_id
            )

            response = tokenizer.decode(
                generate_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens =True
            ).strip()

            print (f"\n {response}")

            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")

            if len(conversation_history) > 8:
                conversation_history = conversation_history[-8:]

        except Exception as e:
            print(f"\n Error: {str(e)}")
            print("Please try again")

if __name__ == "__main__":
    chat_with_cs_chatbot()
