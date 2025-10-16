import os
import json
import time
import ast
from vllm import LLM, SamplingParams
from openai import OpenAI
from utils import search_serper


json_string = """{\"name\": \"Tool name\", \"parameters\": {\"Parameter name\": \"Parameter content\", \"... ...\": \"... ...\"}}
{\"name\": \"... ...\", \"parameters\": {\"... ...\": \"... ...\", \"... ...\": \"... ...\"}}"""

SYS = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
1. Name: Search
Description: Search the answer for a specific query leveraging the google search engine
Parameters: {"query": {"description": "The query to search for.", "type": "string", "default": ""}}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{\"name\": \"Tool name\", \"parameters\": {\"Parameter name\": \"Parameter content\", \"... ...\": \"... ...\"}}
{\"name\": \"... ...\", \"parameters\": {\"... ...\": \"... ...\", \"... ...\": \"... ...\"}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
"""

# USER = """You should perform tool call only if you are not sure about the answer. Otherwise, once you think you can get the final answer, stop calling the tool and give a short answer in the <response> field.\n{query}"""
USER = """{query}\nPerform tool call if you are not sure about the answer. Make a plan to solve the problem step by step."""

JUDGE_SYS = """You are a helpful assistant to help me judge if the answer for a specific query is correct or not. You will be given the query, the ground truth answer and the predicted answer. You should first think about if the predicted answer is semantically the same as the ground truth answer. If so give "Yes" in the field "Judgment", otherwise give "No".

**Output Format**
Your Response:
- Thought: <Your brief thoughts>
- Judgment: <Yes/No>
"""

JUDGE_USER = """**Query**
{query}

**Ground Truth Answer**
{ground_truth}

**Predicted Answer**
{predicted}

Your Response:
"""
        
if os.path.exists("./secret.json"):
    client = OpenAI(
        api_key=json.load(open("./secret.json"))["api_key"],
    )
else:
    client = OpenAI(api_key="sk-...")


def gpt_chatcompletion(messages, model="gpt-4o"):
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                n=1,
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(20)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")


def form_user_prompt(history):
    output = "**Dialogue Records History**\n"
    for h in history:
        if h["role"] == "user":
            output += f"<user> {h['content']} </user>\n\n"
        elif h["role"] == "obs":
            output += f"<obs> {h['content']} </obs>\n\n"
        elif h["role"] == "assistant":
            content = h["content"]
            # if "<think>" in content:
            #     thought = content.split("<think>")[1].split("</think>")[0].strip()
            #     if "<tool_call>" in content:
            #         new_thought = "I should use the appropriate tool with proper parameters to respond to the user's need."
            #     else:
            #         new_thought = "I should directly respond to the user's need."
            #     content = content.replace(thought, new_thought)
            output += f"{content}\n"
    return output.strip()


if __name__ == "__main__":
    # load config
    data_path = "./data.json"
    datas = json.load(open(data_path, "r", encoding="utf-8"))
    
    # add args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, default=None, help="Comma-separated model paths")
    args = parser.parse_args()
    
    # Convert the comma-separated string into a list
    args.model_paths = args.model_paths.split(",") if args.model_paths else []
    args.model_paths = [path.strip() for path in args.model_paths if path.strip()]

    for model_path in args.model_paths:
        model_name = model_path.split("/")[-1].replace("/", "_")
        save_path = f"./output/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        result_save_path = os.path.join(save_path, "result.json")
        
        if not os.path.exists(result_save_path):
            answers = {}
        else:
            answers = json.load(open(result_save_path, "r", encoding="utf-8"))
        
        print("Loading model with vLLM...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=int(os.getenv("WORLD_SIZE", 1)),
            gpu_memory_utilization=0.3,
            max_model_len=4096,
            # dtype="bfloat16"
        )
        sampling_params = SamplingParams(
            max_tokens=4096, 
            temperature=0.001,
        )
        
        log = {"success": 0, "fail": 0, "timeout": 0, "correct": 0, "wrong": 0}
        
        for gold_id, data in enumerate(datas):

            gold_id = str(gold_id)
            if gold_id in answers and answers[gold_id]["score"] == 1:
                continue
                
            print(f" ====== New Test Case: Begin of {gold_id} ====== ")
            
            question = data["Question"]
            gt_answer = data["Answer"]
            
            new_data = {
                "query": question,
                "ground_truth": gt_answer
            }
            
            sys_prompt = SYS
            user_initial_prompt = USER.format(query=question)
            
            history = [{"role": "user", "content": user_initial_prompt}]
            
            round_idx = 0
            final_score = 0
            round_threshold = 4
            while True:
                round_idx += 1
                if round_idx > round_threshold:
                    pd_answer = "Round Limit Exceeded, no answer is derived."
                    break
                
                user_prompt = form_user_prompt(history)
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                result = llm.chat(messages, sampling_params=sampling_params)
                assistant_output = result[0].outputs[0].text.strip()
                
                print(f"\n ====== Round {round_idx} ====== ")
                print(assistant_output)
                
                if "<tool_call>" in assistant_output:
                    try:
                        if "<think>" in assistant_output:
                            thought = assistant_output.split("<think>")[1].split("</think>")[0].strip()
                        else:
                            thought = assistant_output.split("<tool_call>")[0].strip()

                        tool_calls = assistant_output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                        tool_call = tool_calls.split("\n")[0]
                        tool_call = ast.literal_eval(tool_call) # json.loads(tool_call)
                        query = tool_call["parameters"]["query"]
                        search_result = search_serper(query, 5).strip()
                        
                        history.append({"role": "assistant", "content": f"<think> {thought} </think>\n<tool_call>\n{tool_call}\n</tool_call>"})
                        history.append({"role": "obs", "content": f"Search results for the first tool call query:\n{search_result}"})
                        if round_idx < round_threshold - 1:
                            history.append({"role": "user", "content": f"Please continue to search if needed. If you can obtain the answer, please do not call any tool and directly provide a succinct answer in the response."})
                        else:
                            history.append({"role": "user", "content": f"Do not make tool call any more. Please directly provide a succinct answer in your next response."})
                    except:
                        pass
                    # from IPython import embed
                    # embed()
                
                elif "<response>" in assistant_output:
                    if "<think>" in assistant_output:
                        thought = assistant_output.split("<think>")[1].split("</think>")[0].strip()
                    else:
                        thought = assistant_output.split("<response>")[0].strip()

                    response = assistant_output.split("<response>")[1].split("</response>")[0].strip()
                    history.append({"role": "assistant", "content": f"<think> {thought} </think>\n<response> {response} </response>"})
                    pd_answer = response
                    break
                    
                else:
                    if "</think>" in assistant_output:
                        pd_answer = assistant_output.split("</think>")[1].strip()
                    else:
                        pd_answer = assistant_output.strip()
                    # pd_answer = assistant_output
                    history.append({"role": "assistant", "content": f"{assistant_output}"})
                    break
            
            new_data["predicted"] = pd_answer
            new_data["history"] = history
            
            if pd_answer == "Round Limit Exceeded, no answer is derived." or pd_answer == "":
                final_score = 0
                log["timeout"] += 1
            else:
                sys_prompt = JUDGE_SYS
                user_prompt = JUDGE_USER.format(query=question, ground_truth=gt_answer, predicted=pd_answer)
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = gpt_chatcompletion(messages, model="gpt-4o")
                print(" ===== Judgment ===== ")
                print(response)
                thought = response.split("- Thought:")[1].split("- Judgment:")[0].strip()
                judgment = response.split("- Judgment:")[1].strip()
                if judgment == "Yes":
                    final_score = 1
                    log["correct"] += 1
                else:
                    final_score = 0
                    log["wrong"] += 1
            
            new_data["score"] = final_score
            answers[gold_id] = new_data
            
            with open(result_save_path, "w", encoding="utf-8") as f:
                json.dump(answers, f, indent=4, ensure_ascii=False)
            
            log["success"] += 1
            print(log)
            
    leaderboard_path = "./leaderboard.json"
    leaderboard = {}
    result_root = "./output"
    sub_dirs = os.listdir(result_root)
    for dir in sub_dirs:
        score_path = os.path.join(result_root, dir, "result.json")
        if not os.path.exists(score_path):
            continue
        answers = json.load(open(score_path, "r", encoding="utf-8"))
        correct_total = 0
        wrong_total = 0
        tool_total = 0
        for k, v in answers.items():
            if v["score"] == 1:
                correct_total += 1
            else:
                wrong_total += 1
            for h in v["history"]:
                if h["role"] == "obs":
                    tool_total += 1
        # calculate the percentage
        tool_avg = round(tool_total / (correct_total + wrong_total), 2)
        percent = round(correct_total / (correct_total + wrong_total) * 100, 2)
        leaderboard[dir] = {
            "correct": correct_total,
            "wrong": wrong_total,
            "percent": percent,
            "tool_avg": tool_avg,
        }
    
    # sort the leaderboard based on the percent
    sorted_leaderboard = dict(sorted(leaderboard.items(), key=lambda item: item[1]["percent"], reverse=True))
    for key, value in sorted_leaderboard.items():
        print(f"{key} & {value['percent']}\\% & {value['tool_avg']} \\\\")
    
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(sorted_leaderboard, f, indent=4, ensure_ascii=False)
