import os
import json


if __name__ == "__main__":
    # add args
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, default=None, help="Comma-separated model paths")
    args = parser.parse_args()

    # Convert the comma-separated string into a list
    args.model_paths = args.model_paths.split(",") if args.model_paths else []
    args.model_paths = [path.strip() for path in args.model_paths if path.strip()]
    
    for model_path in args.model_paths:
        
        model_name = model_path.split("TinyZero/")[-1].replace("/", "_")
        save_path = f"PATH_TO_YOUR_SCORE_ROOT/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        result_save_path = os.path.join(save_path, "result.json")
        score_save_path = os.path.join(save_path, "score.json")
        
        if os.path.exists(result_save_path):
            results = json.load(open(result_save_path, "r", encoding="utf-8"))
        else:
            assert False, f"No result file found, please check the model path: {model_path}"
        if os.path.exists(score_save_path):
            scores = json.load(open(score_save_path, "r", encoding="utf-8"))
        else:
            scores = {}
        
        for key, result in results.items():
            # if key in scores:
            #     continue
            
            tool_calls = result["tool_calls"]
            answer = result["data"]["answer"]
            if type(answer) == list:
                answer = answer[0]
            answer_name = answer["name"]
            answer_parameters = answer["parameters"]
            
            score = 0
            
            # ================ SOFT MATCH ================
            if len(tool_calls) != 1:
                pass
                # print("Tool calls length is not 1")
            if True:
                try:
                    for tool_call in tool_calls:
                        if type(tool_call) == str:
                            tool_call = json.loads(tool_call)
                        predict = tool_call
                        
                        if "name" not in predict or "parameters" not in predict:
                            # assert False, f"Tool call {tool_call} is not valid in format"
                            name = answer_name
                            parameters = predict
                        else:
                            name = predict["name"]
                            parameters = predict["parameters"]
                        
                        if name == answer_name and parameters == answer_parameters:
                            score += 1
                            break
                
                except Exception as e:
                    print("Error parsing tool call:", tool_call)

            # =============== HARD MATCH ================
            # if len(tool_calls) != 1:
            #     pass
            #     # print("Tool calls length is not 1")
            # else:
            #     try:
            #         tool_call = tool_calls[-1]
            #         if type(tool_call) == str:
            #             tool_call = json.loads(tool_call)
            #         predict = tool_call
                        
            #         name = predict["name"]
            #         parameters = predict["parameters"]
                        
            #         if name == answer_name and parameters == answer_parameters:
            #             score += 1
                
            #     except Exception as e:
            #         print(e)
            #         print("Error parsing tool call:", tool_call)
            
            
            result["score"] = score
            scores[key] = result
                
        with open(score_save_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)
        
        print("All done for", model_path)
        
    # leaderboard
    score_root = "PATH_TO_YOUR_SCORE_ROOT"
    leader_board = {}
    for dir in os.listdir(score_root):
        score_path = os.path.join(score_root, dir, "score.json")
        if not os.path.exists(score_path):
            continue
        record = {
            "correct_lv1": 0,
            "correct_lv2": 0,
            "correct_lv3": 0,
            "total_lv1": 0,
            "total_lv2": 0,
            "total_lv3": 0
        }
        scores = json.load(open(score_path, "r", encoding="utf-8"))
        for key, result in scores.items():
            score = result["score"]
            if score == 1:
                if key.startswith("Level1"):
                    record["correct_lv1"] += 1
                    record["total_lv1"] += 1
                elif key.startswith("Level2"):
                    record["correct_lv2"] += 1
                    record["total_lv2"] += 1
                elif key.startswith("Level3"):
                    record["correct_lv3"] += 1
                    record["total_lv3"] += 1
            else:
                if key.startswith("Level1"):
                    record["total_lv1"] += 1
                elif key.startswith("Level2"):
                    record["total_lv2"] += 1
                elif key.startswith("Level3"):
                    record["total_lv3"] += 1
        
        lv1_acc = round(record["correct_lv1"] / record["total_lv1"] * 100, 2) if record["total_lv1"] > 0 else 0
        lv2_acc = round(record["correct_lv2"] / record["total_lv2"] * 100, 2) if record["total_lv2"] > 0 else 0
        lv3_acc = round(record["correct_lv3"] / record["total_lv3"] * 100, 2) if record["total_lv3"] > 0 else 0
        overall_acc = round((record["correct_lv1"] + record["correct_lv2"] + record["correct_lv3"]) / (record["total_lv1"] + record["total_lv2"] + record["total_lv3"]) * 100, 2) if (record["total_lv1"] + record["total_lv2"] + record["total_lv3"]) > 0 else 0
        
        record["lv1_acc"] = lv1_acc
        record["lv2_acc"] = lv2_acc
        record["lv3_acc"] = lv3_acc
        record["overall_acc"] = overall_acc
        
        leader_board[dir] = record
        
    # sort the leader board based on the overall accuracy
    sorted_leader_board = dict(sorted(leader_board.items(), key=lambda item: item[1]["overall_acc"], reverse=True))
    for key, value in sorted_leader_board.items():
        print(f"{key}: & {value['overall_acc']}\\% & {value['lv1_acc']}\\% & {value['lv2_acc']}\\% & {value['lv3_acc']} \\\\")
    # save the leaderboard
    leaderboard_path = os.path.join(score_root, "leaderboard.json")
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(sorted_leader_board, f, indent=4, ensure_ascii=False)
        print("Leaderboard saved to", leaderboard_path)