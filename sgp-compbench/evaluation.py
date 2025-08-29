import os
import json
import sys
import time
import random
import signal
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
import math
from util import load_json_file, check_svg_validity
from evaluation_prompts import *

def init_worker():
    """
    Initialize worker processes and set signal handling to ensure proper termination
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # set a more aggressive SIGALRM handler to ensure functions don't get stuck
    def timeout_handler(signum, frame):
        raise TimeoutError("Function execution timed out")
    signal.signal(signal.SIGALRM, timeout_handler)

def api_call_4_eval_with_timeout(judge_model, svg_code, output_path, prompt, timeout=60):
    """
    Model API call with strict timeout control
    """
    # set SIGALRM timer to ensure it doesn't get stuck
    signal.alarm(timeout)
    try:
        from util import api_call_4_eval
        result = api_call_4_eval(judge_model, svg_code, output_path=output_path, prompt=prompt)
        # cancel timer
        signal.alarm(0)
        return result
    except Exception as e:
        # cancel timer
        signal.alarm(0)
        print(f"API call exception: {str(e)}")
        raise
    finally:
        # ensure timer is cancelled
        signal.alarm(0)

def get_model_result_safe(args):
    """
    Safe model result retrieval function, suitable for process pools
    """
    judge_model, svg_code, output_path, query, entry_id, timeout = args
    
    print(f"Processing ID: {entry_id} with timeout {timeout}s")
    start_time = time.time()
    
    try:
        # use API call with timeout
        result = api_call_4_eval_with_timeout(judge_model, svg_code, output_path, query, timeout)
        
        if result.get('success'):
            response = result.get('response')
            score = None
            
            # extract score logic is the same as the original code
            if "SCORE:" in response:
                try:
                    score_str = response.split("SCORE:")[1].strip().split()[0]
                    if score_str in ['100', '50', '30', '0']:
                        score = score_str
                except Exception:
                    score = None
                    
            if score is None:
                import re
                matches = re.findall(r'(\d+)', response)
                allowed_scores = ['100', '50', '30', '0']
                filtered = [m for m in matches if m in allowed_scores]
                if filtered:
                    score = filtered[-1]
                    
            if score is not None:
                elapsed = time.time() - start_time
                print(f"ID: {entry_id}, Completed in {elapsed:.2f}s with score {score}")
                return {"id": entry_id, "score": int(score), "processing_time": elapsed}
            else:
                return {"id": entry_id, "error": "Failed to extract score", "score": 0}
        else:
            return {"id": entry_id, "error": "API call failed", "score": 0}
            
    except TimeoutError:
        print(f"ID: {entry_id}, Timed out after {timeout}s")
        return {"id": entry_id, "error": f"Timeout after {timeout}s", "score": 0}
    except Exception as e:
        print(f"ID: {entry_id}, Error: {str(e)}")
        return {"id": entry_id, "error": str(e), "score": 0}

def process_entry(args):
    """
    Process a single entry and return the result (including error handling)
    """
    judge_model, item, bench_data, timeout = args
    entry_id = item.get('id', 'unknown')
    result = {"id": entry_id}
    
    try:
        # set timeout in the process
        signal.alarm(timeout)
        
        # find matching benchmark item
        found_bench_item = False
        for bench_item in bench_data:
            if bench_item.get('id') == entry_id:
                found_bench_item = True
                # copy relevant fields
                for field in ['prompt', 'gen_type', 'major_type', 'minor_type', 'metadata']:
                    if field in bench_item:
                        result[field] = bench_item.get(field, '')
                
                # process different types of prompts
                major_type = bench_item.get('major_type', '')
                metadata = bench_item.get('metadata', {})
                
                if major_type == "binding":
                    result["relation"] = metadata.get('relation', '')
                    result["nouns"] = metadata.get('nouns', [])
                    if bench_item.get('minor_type') == "texture":
                        result["textures"] = metadata.get('textures', [])
                    if bench_item.get('minor_type') == "color":
                        result["colors"] = metadata.get('colors', [])
                    if bench_item.get('minor_type') == "shape":
                        result["shapes"] = metadata.get('shapes', [])
                
                elif major_type == "relation":
                    result["relation"] = metadata.get('relation', '')
                    result["nouns"] = metadata.get('nouns', [])
                
                elif major_type == "numeracy":
                    result["total_count"] = metadata.get('total_count', '')
                    object_counts = metadata.get('object_counts', [])
                    result["object_counts"] = object_counts
                    
                    # extract nouns and counts
                    nouns = []
                    counts = []
                    for entry in object_counts:
                        nouns.append(entry.get('noun', ''))
                        counts.append(entry.get('count', ''))
                    result["nouns"] = nouns
                    result["counts"] = counts
                
                break
        
        if not found_bench_item:
            result["error"] = f"No matching bench item found for ID: {entry_id}"
            result["score"] = 0
            signal.alarm(0)  # cancel timeout
            return result
        
        # check SVG validity
        svg_code = item.get('svg', '')
        output_path = f"output_{entry_id}.png"
        
        # import and use check_svg_validity function
        is_valid, error_msg = check_svg_validity(svg_code)
        result["valid_svg"] = is_valid
        if not is_valid:
            result["svg_error"] = error_msg
            result["score"] = 0
            signal.alarm(0)  # cancel timeout
            return result
        
        # evaluate SVG based on type
        major_type = result.get("major_type")
        prompt = result.get("prompt", "")
        
        if major_type == "binding":
            query = eval_prompt_binding(prompt)
            score_result = get_model_result_safe((judge_model, svg_code, output_path, query, entry_id, timeout//2))
            result["score"] = score_result.get("score", 0)
            
        elif major_type == "relation":
            query = eval_prompt_relation(prompt)
            score_result = get_model_result_safe((judge_model, svg_code, output_path, query, entry_id, timeout//2))
            result["score"] = score_result.get("score", 0)
            
        elif major_type == "numeracy":
            # execute three parts of numeracy evaluation
            component_scores = {}
            
            # total count (weight 0.2)
            total_count = result["total_count"]
            total_count_query = eval_prompt_numeracy_total(total_count)
            total_score_result = get_model_result_safe(
                (judge_model, svg_code, output_path, total_count_query, f"{entry_id}_total", timeout//4)
            )
            total_count_score = total_score_result.get("score", 0)
            component_scores["total_count_score"] = total_count_score
            
            # item (weight 0.2)
            nouns = result["nouns"]
            item_query = eval_prompt_numeracy_item(nouns)
            item_score_result = get_model_result_safe(
                (judge_model, svg_code, output_path, item_query, f"{entry_id}_item", timeout//4)
            )
            item_score = item_score_result.get("score", 0)
            component_scores["item_score"] = item_score
            
            # process count binding (weight 0.6)
            count_binding_scores = []
            nouns = result["nouns"]
            counts = result["counts"]
            
            for i in range(min(len(nouns), len(counts))):
                count_binding_query = eval_prompt_numeracy_count_binding(counts[i], nouns[i])
                binding_result = get_model_result_safe(
                    (judge_model, svg_code, output_path, count_binding_query, f"{entry_id}_binding_{i}", timeout//4)
                )
                count_binding_scores.append(binding_result.get("score", 0))
            
            mean_binding_score = sum(count_binding_scores) / len(count_binding_scores) if count_binding_scores else 0
            component_scores["mean_count_binding_score"] = mean_binding_score
            
            # calculate final score
            final_score = int(0.2 * float(total_count_score) + 0.2 * float(item_score) + 0.6 * float(mean_binding_score))
            result["score"] = final_score
            result["component_scores"] = component_scores
            
        else:
            # unknown type
            result["score"] = 0
            result["error"] = f"Unknown major_type: {major_type}"
        
        # cancel timeout signal
        signal.alarm(0)
        return result
        
    except TimeoutError:
        # if the entire process times out
        print(f"ID: {entry_id}, Complete process timed out")
        result["error"] = f"Complete process timed out after {timeout}s"
        result["score"] = 0
        return result
        
    except Exception as e:
        # capture all other exceptions
        print(f"ID: {entry_id}, Error in process_entry: {str(e)}")
        result["error"] = str(e)
        result["score"] = 0
        return result
    
    finally:
        # ensure timeout signal is cancelled
        signal.alarm(0)

def process_batch(batch_data, judge_model, bench_data, max_workers=4, timeout=300):
    """
    Process a batch of data using multiple processes
    """
    # prepare parameters
    process_args = [(judge_model, item, bench_data, timeout) for item in batch_data]
    
    # shuffle processing order to avoid same entries always getting stuck at the end
    random.shuffle(process_args)
    
    results = []
    processed_ids = set()
    
    # use process pool to process
    with Pool(processes=max_workers, initializer=init_worker) as pool:
        try:
            # submit all tasks and get iterator
            result_iter = pool.imap_unordered(process_entry, process_args)
            
            # use tqdm to track progress
            with tqdm(total=len(batch_data), desc="Processing entries", unit="entry") as pbar:
                # set global timeout
                start_time = time.time()
                global_timeout = timeout * 2  # overall timeout is twice the single entry timeout
                
                for i, result in enumerate(result_iter):
                    # check if global timeout is exceeded
                    if time.time() - start_time > global_timeout:
                        print(f"Global timeout reached after processing {i} entries")
                        pool.terminate()  # force terminate all worker processes
                        break
                    
                    # add result and update progress bar
                    if result:
                        results.append(result)
                        entry_id = result.get('id')
                        if entry_id:
                            processed_ids.add(entry_id)
                        print(f"Completed {i+1}/{len(batch_data)}: ID={entry_id}, Score={result.get('score', 'N/A')}")
                    pbar.update(1)
                    
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        except Exception as e:
            print(f"Error in process_batch: {str(e)}")
            pool.terminate()
        finally:
            pool.close()
            pool.join()
    
    # find unprocessed entries
    all_ids = {item.get('id') for item in batch_data if item.get('id')}
    unprocessed_ids = all_ids - processed_ids
    unprocessed_entries = [item for item in batch_data if item.get('id') in unprocessed_ids]
    
    return {
        "results": results,
        "unprocessed_entries": unprocessed_entries,
        "processed_count": len(results),
        "unprocessed_count": len(unprocessed_entries)
    }

def process_file(input_file, bench_file, judge_model, output_file=None, max_workers=4, 
                batch_size=50, timeout=300):
    """
    Process all entries in the input file and save the results
    """
    print(f"Loading data from {input_file} and {bench_file}")
    input_data = load_json_file(input_file)
    bench_data = load_json_file(bench_file)

    dir_name = "evaluation_results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    if not output_file:
        timestamp = int(time.time())
        output_file = f"{dir_name}/result_{os.path.basename(input_file).replace('.json', '')}_{timestamp}.json"
    
    # process in batches
    all_results = []
    all_unprocessed = []
    
    num_batches = math.ceil(len(input_data) / batch_size)
    print(f"Processing {len(input_data)} entries in {num_batches} batches of {batch_size}")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(input_data))
        batch = input_data[start_idx:end_idx]
        
        print(f"\nProcessing batch {i+1}/{num_batches} ({start_idx}-{end_idx-1})")
        batch_result = process_batch(batch, judge_model, bench_data, max_workers, timeout)
        
        all_results.extend(batch_result["results"])
        all_unprocessed.extend(batch_result["unprocessed_entries"])
        
        print(f"Batch {i+1} completed: {batch_result['processed_count']} processed, "
              f"{batch_result['unprocessed_count']} unprocessed")
        
        # save interim results after each batch
        interim_output = {
            "results": all_results,
            "unprocessed_entries": all_unprocessed,
            "interim_batch": i+1,
            "total_batches": num_batches,
            "processed_count": len(all_results),
            "unprocessed_count": len(all_unprocessed)
        }
        
        interim_file = f"{dir_name}/interim_{os.path.basename(output_file)}"
        with open(interim_file, 'w') as f:
            json.dump(interim_output, f, indent=2)
        print(f"Saved interim results to {interim_file}")
    
    # calculate statistics
    valid_count = sum(1 for r in all_results if r.get('valid_svg', False))
    score_counts = {'0': 0, '30': 0, '50': 0, '100': 0}
    
    for result in all_results:
        score = str(result.get('score', 0))
        if score in score_counts:
            score_counts[score] += 1
    
    # prepare final output
    final_output = {
        "results": all_results,
        "unprocessed_entries": all_unprocessed,
        "stats": {
            "total_input": len(input_data),
            "processed": len(all_results),
            "unprocessed": len(all_unprocessed),
            "valid_svg": valid_count,
            "score_distribution": score_counts
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\nProcessing completed. Results saved to {output_file}")
    print(f"Processed {len(all_results)}/{len(input_data)} entries ({len(all_unprocessed)} unprocessed)")
    print(f"Valid SVGs: {valid_count}")
    print(f"Score distribution: {score_counts}")
    
    # if there are unprocessed entries, save them to a separate file
    if all_unprocessed:
        unprocessed_file = f"{dir_name}/remaining_{os.path.basename(input_file)}"
        with open(unprocessed_file, 'w') as f:
            json.dump(all_unprocessed, f, indent=2)
        print(f"Saved {len(all_unprocessed)} unprocessed entries to {unprocessed_file}")
    
    return final_output

if __name__ == "__main__":
    # set parameters
    # Directly assign parameters instead of using argparse
    # Please modify these variables as needed
    args = type('Args', (object,), {})()
    args.input = "svg_generation_results/o4-mini_all_result.json"  # Input JSON file with SVG entries
    args.bench = "prompts/svg-compbench-prompts.json"  # Benchmark JSON file with prompts
    args.model = "google/gemini-2.5-flash-preview"  # Judge model to use
    args.output = "evaluation_results/o4-mini_all_result.json"  # Output JSON file (default: auto-generated)
    args.workers = 100  # Number of worker processes (default: 4)
    args.batch = 50  # Batch size for processing (default: 50)
    args.timeout = 20  # Timeout in seconds for each entry (default: 300)

    process_file(
        args.input,
        args.bench,
        args.model,
        args.output,
        args.workers,
        args.batch,
        args.timeout
    )