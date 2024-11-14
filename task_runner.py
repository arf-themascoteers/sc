import torch
from ds_manager import DSManager
from reporter import Reporter
import pandas as pd
from metrics import Metrics
from algorithm import Algorithm
import train_test_evaluator


class TaskRunner:
    def __init__(self, task, tag="results", verbose=False, remove_bg=False, split="bsnet"):
        torch.manual_seed(3)
        self.task = task
        self.verbose = verbose
        self.remove_bg = remove_bg
        self.tag = tag
        self.reporter = Reporter(self.tag)
        self.cache = pd.DataFrame(columns=["dataset","algorithm","cache_tag","oa","aa","k","time","selected_bands","selected_weights"])
        self.split = split

    def evaluate(self):
        for dataset_name in self.task["datasets"]:
            dataset = DSManager(name=dataset_name, split=self.split)
            for index, algorithm in enumerate(self.task["algorithms"]):
                for target_size in self.task["target_sizes"]:
                    print(dataset_name, algorithm, target_size)
                    algorithm_object = Algorithm.create(algorithm, target_size, dataset, self.tag, self.reporter, self.verbose)
                    self.process_a_case(algorithm_object)

        self.reporter.save_results()
        return self.reporter.get_summary(), self.reporter.get_details()

    def process_a_case(self, algorithm:Algorithm):
        metric = self.reporter.get_saved_metrics(algorithm)
        if metric is None:
            oas, aas, ks, metric = self.get_results_for_a_case(algorithm)
            self.reporter.write_summary(algorithm, oas, aas, ks, metric)
        else:
            print(algorithm.get_name(), "for", algorithm.dataset.get_name(), "for",
                  algorithm.target_size,"was done. Skipping")

    def get_results_for_a_case(self, algorithm:Algorithm):
        metric = self.get_from_cache(algorithm)
        if metric is not None:
            print(f"Selected features got from cache for {algorithm.dataset.get_name()} "
                  f"for size {algorithm.target_size} "
                  f"for {algorithm.get_name()} "
                  f"for cache_tag {algorithm.get_cache_tag()}")
            algorithm.set_selected_indices(metric.selected_bands)
            algorithm.set_weights(metric.selected_weights)
            return algorithm.compute_performance()
        print(f"NOT FOUND in cache for {algorithm.dataset.get_name()} "
              f"for size {algorithm.target_size} "
              f"for {algorithm.get_name()} "
              f"for cache_tag {algorithm.get_cache_tag()}. Computing.")
        oas, aas, ks, metric = algorithm.compute_performance()
        self.save_to_cache(algorithm, metric)
        return oas, aas, ks, metric

    def save_to_cache(self, algorithm:Algorithm, metric:Metrics):
        if not algorithm.is_cacheable():
            return
        self.cache.loc[len(self.cache)] = {
            "dataset":algorithm.dataset.get_name(),
            "algorithm": algorithm.get_name(),
            "cache_tag": algorithm.get_cache_tag(),
            "time":metric.time,"oa":metric.oa,"aa":metric.aa,"k":metric.k,
            "selected_bands":algorithm.get_all_indices(),
            "selected_weights":algorithm.get_weights()
        }

    def get_from_cache(self, algorithm:Algorithm):
        if not algorithm.is_cacheable():
            return None
        if len(self.cache) == 0:
            return None
        rows = self.cache.loc[
            (self.cache["dataset"] == algorithm.dataset.get_name()) &
            (self.cache["algorithm"] == algorithm.get_name()) &
            (self.cache["cache_tag"] == algorithm.get_cache_tag())
        ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        selected_bands = row["selected_bands"][0:algorithm.target_size]
        selected_weights = row["selected_weights"][0:algorithm.target_size]
        return Metrics(row["time"], row["oa"],row["aa"], row["k"], selected_bands, selected_weights)

    def evaluate_for_all_features(self, dataset):
        for fold, (train_x, test_x, train_y, test_y) in enumerate(dataset.get_k_folds()):
            oa, aa, k = train_test_evaluator.evaluate_split(train_x, test_x, train_y, test_y, classification=dataset.is_classification())
            self.reporter.write_details_all_features(fold, dataset.get_name(), oa, aa, k)


