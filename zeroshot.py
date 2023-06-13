import argparse
import json
import logging
import fnmatch

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='For OPT model to load; pass `facebook/opt-X`.\\ `llama hf path/X`.'
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed for sampling the calibration data.')
    parser.add_argument(
        '--load', type=str, default='',
        help='Load fake quantized model.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--logfile', type=str, default='',
        help='Logging file name'
    )

    # parser.add_argument("--model", required=True)
    # parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1) # in gptq, BS=1 is used for zeroShot tasks!
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true") # in gptq, no_cache = True
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    import random
    import os
    import numpy as np
    import torch
    import time

    args = parse_args()
    
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    # description_dict = {}
    # if args.description_dict_path:
    #     with open(args.description_dict_path, "r") as f:
    #         description_dict = json.load(f)

    def seed_all(seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    seed_all(args.seed)

    # args.batch_size = 1
    args.no_cache = True
    args.device = torch.device('cuda:0')
    tick = time.time()
    results = evaluator.simple_evaluate(
        model=args.model,
        load=args.load,
        args=args,
        # model_args=args.model_args,
        tasks=task_names,
        # num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        # limit=args.limit,
        # description_dict=description_dict,
        # decontamination_ngrams_path=args.decontamination_ngrams_path,
        # check_integrity=args.check_integrity,
    )
    dumped = json.dumps(results, indent=2)
    print(dumped)
    t = time.time() - tick
    print("total time :",t)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.load if args.load else args.model} batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))
    if args.logfile:
        with open(args.logfile,'a') as fp:
            fp.write(f"model : {results['config']['model']} | batch_size : {results['config']['batch_size']} | evaluate time : {t}\n")
            fp.write(evaluator.make_table(results))
            # for task in results['results'].keys():
            #     fp.write(f"{task} | {results['results'][task]}\n")
            fp.write('\n')



if __name__ == "__main__":
    main()