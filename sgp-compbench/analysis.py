import json
import os
import argparse
from collections import defaultdict, Counter

def load_json_file(file_path):
    """load json file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_results(result_file, output_dir=None, analyze_numeracy=False):
    """analyze the result file, count the score by category"""
    print(f"analyze file: {result_file}")
    
    # load the result data
    data = load_json_file(result_file)
    
    # determine the storage directory
    if output_dir is None:
        output_dir = "analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get all the results
    results = data.get("results", [])
    if not results:
        print("warning: no result data found!")
        return
    
    print(f"total {len(results)} result records")
    
    # group by major type(major_type)
    results_by_type = defaultdict(list)
    for result in results:
        major_type = result.get("major_type", "unknown")
        results_by_type[major_type].append(result)
    
    # initialize the statistics data
    stats = {
        "total_count": len(results),
        "by_major_type": {},
        "score_distribution": Counter(),
        "valid_svg_count": sum(1 for r in results if r.get("valid_svg", False)),
        "invalid_svg_count": sum(1 for r in results if not r.get("valid_svg", False)),
        "total_score": 0,
        "detailed_scores": {
            "attribute_binding": {
                "color": 0,
                "shape": 0,
                "texture": 0,
                "total": 0
            },
            "relation": {
                "2d": 0,
                "3d": 0,
                "implicit": 0,
                "total": 0
            },
            "numeracy": {
                "total_count_score": 0,
                "item_score": 0,
                "mean_count_binding_score": 0,
                "total": 0
            }
        }
    }
    
    # calculate the statistics data for each type
    for major_type, type_results in results_by_type.items():
        # basic count
        total = len(type_results)
        valid_count = sum(1 for r in type_results if r.get("valid_svg", False))
        
        # count the score
        scores = [r.get("score", 0) for r in type_results]
        score_counts = Counter(scores)
        avg_score = sum(scores) / total if total > 0 else 0
        
        # further divide by minor_type
        minor_types = defaultdict(list)
        for r in type_results:
            minor_type = r.get("minor_type", "unknown")
            minor_types[minor_type].append(r)
        
        # calculate the average score for each minor_type
        minor_type_stats = {}
        for minor_type, minor_results in minor_types.items():
            minor_scores = [r.get("score", 0) for r in minor_results]
            minor_avg = sum(minor_scores) / len(minor_results) if minor_results else 0
            minor_type_stats[minor_type] = {
                "count": len(minor_results),
                "average_score": minor_avg,
                "score_distribution": dict(Counter(minor_scores))
            }
            
            # collect the detailed score for specific sub-categories
            if major_type == "binding" and minor_type in ["color", "shape", "texture"]:
                stats["detailed_scores"]["attribute_binding"][minor_type] = minor_avg
                    
            elif major_type == "relation":
                if minor_type == "2d_spatial":
                    stats["detailed_scores"]["relation"]["2d"] = minor_avg
                elif minor_type == "3d_spatial":
                    stats["detailed_scores"]["relation"]["3d"] = minor_avg
                elif minor_type == "implicit":
                    stats["detailed_scores"]["relation"]["implicit"] = minor_avg
        
        # save the statistics data for this type
        stats["by_major_type"][major_type] = {
            "count": total,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "valid_percentage": (valid_count / total * 100) if total > 0 else 0,
            "average_score": avg_score,
            "score_distribution": dict(score_counts),
            "by_minor_type": minor_type_stats
        }
        
        # calculate the total score for the major type
        if major_type == "binding":
            stats["detailed_scores"]["attribute_binding"]["total"] = avg_score
        elif major_type == "relation":
            stats["detailed_scores"]["relation"]["total"] = avg_score
        elif major_type == "numeracy":
            stats["detailed_scores"]["numeracy"]["total"] = avg_score
            
            # calculate the component score for numeracy
            numeracy_component_scores = {
                "total_count_score": [],
                "item_score": [],
                "mean_count_binding_score": []
            }
            
            for result in type_results:
                component_scores = result.get("component_scores", {})
                if component_scores:
                    for key in numeracy_component_scores.keys():
                        value = component_scores.get(key)
                        if value is not None:
                            # handle different data types
                            if isinstance(value, str):
                                try:
                                    value = float(value)
                                except:
                                    pass
                            numeracy_component_scores[key].append(value)
            
            # calculate the average score
            for key, values in numeracy_component_scores.items():
                if values:
                    stats["detailed_scores"]["numeracy"][key] = sum(values) / len(values)
        
        # add to the overall score distribution
        stats["score_distribution"].update(scores)
    
    # calculate the overall average score
    all_scores = [r.get("score", 0) for r in results]
    stats["total_score"] = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # save the statistics data to JSON
    stats_file = os.path.join(output_dir, f"{os.path.basename(result_file).replace('.json', '')}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"statistics data saved to: {stats_file}")
    
    # print the simplified key indicators
    print("\nEvaluation Result Summary:")
    print(f"Total Average Score: {stats['total_score']:.2f}")
    
    # attribute binding score
    attr_scores = stats["detailed_scores"]["attribute_binding"]
    print("\nAttribute Binding Score:")
    print(f"  Color: {attr_scores['color']:.2f}")
    print(f"  Shape: {attr_scores['shape']:.2f}")
    print(f"  Texture: {attr_scores['texture']:.2f}")
    print(f"  Total: {attr_scores['total']:.2f}")
    
    # relation score
    rel_scores = stats["detailed_scores"]["relation"]
    print("\nRelation Score:")
    print(f"  2D: {rel_scores['2d']:.2f}")
    print(f"  3D: {rel_scores['3d']:.2f}")
    print(f"  Implicit: {rel_scores['implicit']:.2f}")
    print(f"  Total: {rel_scores['total']:.2f}")
    
    # numeracy score
    num_scores = stats["detailed_scores"]["numeracy"]
    print("\nNumeracy Score:")
    print(f"  Total Count Score: {num_scores['total_count_score']:.2f}")
    print(f"  Item Score: {num_scores['item_score']:.2f}")
    print(f"  Mean Count Binding Score: {num_scores['mean_count_binding_score']:.2f}")
    print(f"  Total: {num_scores['total']:.2f}")
    
    # if need to analyze numeracy but not print the detailed results
    if analyze_numeracy and "numeracy" in results_by_type:
        analyze_numeracy_details(results_by_type["numeracy"], result_file, output_dir, print_results=True)
    
    return stats

def analyze_numeracy_details(numeracy_results, result_file, output_dir, print_results=True):
    """analyze the score distribution of numeracy type, by total_count and component_scores"""
    
    # group by total_count
    results_by_total_count = defaultdict(list)
    for result in numeracy_results:
        total_count = result.get("total_count")
        if total_count is not None:
            # handle different data types
            if isinstance(total_count, str):
                total_count = int(total_count)
            results_by_total_count[total_count].append(result)
    
    # initialize the statistics data
    stats = {
        "total_numeracy_count": len(numeracy_results),
        "by_total_count": {},
        "component_scores": {
            "total_count_score": [],
            "item_score": [],
            "mean_count_binding_score": []
        },
        "component_score_distribution": {
            "total_count_score": Counter(),
            "item_score": Counter(),
            "mean_count_binding_score": Counter()
        },
        "valid_svg_count": sum(1 for r in numeracy_results if r.get("valid_svg", False))
    }
    
    # calculate the statistics data for each total_count
    if print_results:
        print("count by total_count:")
    
    for total_count, count_results in sorted(results_by_total_count.items()):
        # basic count
        total = len(count_results)
        valid_count = sum(1 for r in count_results if r.get("valid_svg", False))
        
        # count the score
        scores = [r.get("score", 0) for r in count_results]
        score_counts = Counter(scores)
        avg_score = sum(scores) / total if total > 0 else 0
        
        # collect the component score
        component_scores_list = {
            "total_count_score": [],
            "item_score": [],
            "mean_count_binding_score": []
        }
        
        for result in count_results:
            component_scores = result.get("component_scores", {})
            if component_scores:
                for key in component_scores_list.keys():
                    value = component_scores.get(key)
                    if value is not None:
                        # handle different data types
                        if isinstance(value, str):
                            try:
                                value = float(value)
                            except:
                                pass
                        component_scores_list[key].append(value)
                        stats["component_scores"][key].append(value)
                        stats["component_score_distribution"][key][value] += 1
        
        # calculate the average score for the component
        component_avgs = {}
        for key, values in component_scores_list.items():
            if values:
                component_avgs[key] = sum(values) / len(values)
            else:
                component_avgs[key] = 0
        
        # save the statistics data for this type
        stats["by_total_count"][total_count] = {
            "count": total,
            "valid_count": valid_count,
            "valid_percentage": (valid_count / total * 100) if total > 0 else 0,
            "average_score": avg_score,
            "score_distribution": dict(score_counts),
            "component_averages": component_avgs
        }
        
        if print_results:
            print(f"Total Count = {total_count}:")
            print(f"  Count: {total}")
            print(f"  Valid SVG: {valid_count} ({stats['by_total_count'][total_count]['valid_percentage']:.2f}%)")
            print(f"  Average Score: {avg_score:.2f}")
            print(f"  Component Averages:")
            for comp_key, comp_avg in component_avgs.items():
                print(f"    - {comp_key}: {comp_avg:.2f}")
    
    # calculate the overall average score for the component
    for key, values in stats["component_scores"].items():
        if values:
            stats[f"{key}_average"] = sum(values) / len(values)
        else:
            stats[f"{key}_average"] = 0
    
    # save the statistics data to JSON
    stats_file = os.path.join(output_dir, f"{os.path.basename(result_file).replace('.json', '')}_numeracy_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if print_results:
        print(f"\nstatistics data saved to: {stats_file}")
    
    return stats