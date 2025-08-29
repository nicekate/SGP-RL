from util import load_json_file, api_call_4_generation
import concurrent.futures
from tqdm import tqdm
import json
import os

def get_svg_generation_prompt(description):
    """
    Generate the main SVG generation prompt given a description.
    """
    return (
        "You are an expert in generating SVG code.\n"
        "Your task is to carefully analyze the description and produce only the corresponding SVG code.\n"
        "Do not generate any images or explanations—output strictly the SVG code that fulfills the following description.\n\n"
        "You must generate the SVG code in the following format:\n"
        "1. Start with <svg> tag\n"
        "2. Include all necessary SVG elements and attributes\n"
        "3. End with </svg> tag\n\n"
        "You must ensure that the SVG code is valid and complete.\n\n"
        "You must ensure the SVG code is minimal and only includes elements necessary to satisfy the description.\n\n"
        
        f"Description: {description}"
    )

def parse_svg_from_response(response):
    """
    Extract SVG code from the model response and indicate success.
    
    Returns:
        tuple: (svg_code, success)
            svg_code (str): The extracted SVG code, or empty string if not found.
            success (bool): True if SVG code was successfully extracted, False otherwise.
    """
    svg_start = response.find('<svg')
    svg_end = response.find('</svg>')
    if svg_start == -1 or svg_end == -1:
        return "", False  # Extraction failed
    svg_end += len('</svg>')
    return response[svg_start:svg_end], True  # Extraction succeeded

def reprocess_failed_entries(model, result_path, raw_result_path, max_retry=3, max_workers=100):
    """
    Read the result JSON file, identify entries where success is false,
    and try to regenerate SVGs for those entries using parallel processing.
    
    Args:
        model: The model to use for generation
        result_path: Path to the result JSON file
        raw_result_path: Path to the raw result JSON file
        max_retry: Maximum number of retry attempts per entry
        max_workers: Maximum number of parallel workers
        
    Returns:
        tuple: (updated_results, updated_raw_results)
    """
    # Load existing results
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    with open(raw_result_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)
    
    # Find failed entries
    failed_entries = []
    for entry in results:
        if not entry.get("success", False):
            failed_entries.append(entry)
    
    if not failed_entries:
        print("No failed entries found. All SVGs were generated successfully.")
        return results, raw_results
    
    print(f"Found {len(failed_entries)} failed entries. Attempting to regenerate in parallel...")
    
    # Build a map of id to index for updating entries
    result_id_to_index = {entry["id"]: i for i, entry in enumerate(results)}
    raw_result_id_to_index = {entry["id"]: i for i, entry in enumerate(raw_results)}
    
    # Load original prompts
    benchmark_path = "prompts/svg-compbench-prompts.json"
    prompts = load_json_file(benchmark_path)
    id_to_prompt = {entry.get("id"): entry.get("prompt", "") for entry in prompts}
    
    # Define a function to process a single failed entry
    def process_failed_entry(failed_entry):
        entry_id = failed_entry["id"]
        
        if entry_id not in id_to_prompt:
            print(f"Warning: Prompt for ID {entry_id} not found. Skipping.")
            return {"id": entry_id, "success": False, "svg": "", "raw_response": ""}
        
        entry_prompt = id_to_prompt[entry_id]
        full_prompt = get_svg_generation_prompt(entry_prompt)
        
        for attempt in range(max_retry):
            response = api_call_4_generation(full_prompt, model, 20000)
            # response = call_model_api(model, full_prompt, 40000)
            svg, success = parse_svg_from_response(response)
            print(f"Retry ID: {entry_id}, Attempt: {attempt+1}, Success: {success}")
            
            if success:
                return {"id": entry_id, "svg": svg, "raw_response": response, "success": True}
        
        # If all retries failed, return last attempt
        return {"id": entry_id, "svg": svg, "raw_response": response, "success": False}
    
    # Use ThreadPoolExecutor for parallel processing
    updated_entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(failed_entries))) as executor:
        futures = {executor.submit(process_failed_entry, entry): entry["id"] for entry in failed_entries}
        with tqdm(total=len(failed_entries), desc="Regenerating failed SVGs") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                updated_entries.append(result)
                pbar.update(1)
    
    # Update the results and raw_results with the new values
    for entry in updated_entries:
        entry_id = entry["id"]
        if entry_id in result_id_to_index:
            # Update regular results
            results[result_id_to_index[entry_id]] = {
                "id": entry_id, 
                "svg": entry["svg"], 
                "success": entry["success"]
            }
            
            # Update raw results
            raw_results[raw_result_id_to_index[entry_id]] = {
                "id": entry_id, 
                "svg": entry["svg"], 
                "raw_response": entry["raw_response"], 
                "success": entry["success"]
            }
    
    # Save updated results
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(raw_result_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    
    # Count remaining failures
    remaining_failures = sum(1 for entry in results if not entry.get("success", False))
    print(f"Regeneration complete. Remaining failures: {remaining_failures}")
    
    return results, raw_results

def generate_all_svg(model, bench_path):
    """
    Generate all SVGs for the given model.
    """

    MAX_WORKERS = 100
    max_tokens = 10000
    model = model
    output_dir = "svg_generation_results"
    os.makedirs(output_dir, exist_ok=True)

    benchmark_path = bench_path
    # load all prompts
    prompts = load_json_file(benchmark_path)
    test_samples = prompts

    results = []
    raw_results = []

    def process_entry(entry, max_retry=3):
        """
        Generate a single SVG for the given entry.
        If the parsing fails, it will be retried up to max_retry times.
        """
        entry_id = entry.get("id")
        entry_prompt = entry.get("prompt", "")
        full_prompt = get_svg_generation_prompt(entry_prompt)
        for attempt in range(max_retry):
            # Call the API to get the raw response
            response = api_call_4_generation(full_prompt, model, max_tokens)
            # response = call_model_api(model, full_prompt, max_tokens)
            svg, success = parse_svg_from_response(response)
            print(f"ID: {entry_id}, Success: {success}")
            if success:
                # Return result including parse success
                return {"id": entry_id, "svg": svg, "raw_response": response, "success": success}
        # If all retries failed, return the last response and mark as failed
        return {"id": entry_id, "svg": svg, "raw_response": response, "success": False}

    # Use ThreadPoolExecutor to generate SVGs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(prompts))) as executor:
        futures = {executor.submit(process_entry, entry): entry.get("id") for entry in test_samples}
        with tqdm(total=len(prompts), desc="Generating SVGs (all)") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                # only save id, svg, success
                results.append({"id": result["id"], "svg": result["svg"], "success": result["success"]})
                # save raw information
                raw_results.append(result)
                pbar.update(1)

    # save results to JSON file (only id, svg, success)
    model_name = model.split("/")[-1]
    result_path = os.path.join(output_dir, f"{model_name}_all_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # save raw results to a separate JSON file (id, svg, raw_response, success)
    raw_result_path = os.path.join(output_dir, f"{model_name}_all_raw_result.json")
    with open(raw_result_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    
    # check if there are any failed entries, if so, try to regenerate them
    failed_count = sum(1 for entry in results if not entry.get("success", False))
    if failed_count > 0:
        print(f"Found {failed_count} failed SVG generations. Attempting to regenerate...")
        reprocess_failed_entries(model, result_path, raw_result_path, max_retry=3)

if __name__ == "__main__":
    generate_all_svg("model_name")